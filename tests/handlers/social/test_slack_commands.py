"""Tests for Slack slash command implementations.

Covers all routes and behavior of the CommandsMixin used by SlackHandler:
- _handle_slash_command dispatch (help, status, debate, plan, implement,
  gauntlet, ask, search, leaderboard, recent, agents, unknown)
- Per-workspace rate limiting
- Per-user rate limiting
- Audit logging on success, error, and rate limit
- _command_help
- _command_status (ELO import success, ImportError, data errors, connection errors)
- _command_agents (sorted by ELO, empty list, errors)
- _command_ask (validation, async queueing, too short, too long)
- _command_search (db.search, db.list fallback, empty query, no results, errors)
- _command_leaderboard (table rendering, empty, errors)
- _command_recent (dict + object formats, buttons, empty, errors)
- _command_debate (validation, plan mode, implement mode, short/long topic)
- _command_gauntlet (validation, short/long statement, async queueing)
- _answer_question_async (happy path, fallback to debate, error path)
- _run_gauntlet_async (happy path with vulns, error from API, exception)
- _create_debate_async (full flow, no agents, error)
- _update_debate_status (store available, store unavailable)
- Error handling for ValueError/KeyError/etc in slash command dispatch
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
    """Disable the @rate_limit decorator so it does not interfere with tests.

    The rate_limit decorator on _handle_slash_command is applied at import time.
    We bypass it by patching the rate limiter to always allow requests.
    """
    try:
        from aragora.server.handlers.utils import rate_limit as rl_mod

        monkeypatch.setattr(rl_mod, "_RATE_LIMIT_DISABLED", True, raising=False)
    except (ImportError, AttributeError):
        pass
    yield


@pytest.fixture
def mock_handler_obj():
    """Create a mock HTTP handler object with Slack body attributes."""

    def _make(form_data: dict[str, str] | None = None, **attrs):
        h = MagicMock()
        body_str = _make_form_body(**(form_data or {}))
        h._slack_body = body_str
        h._slack_workspace = None
        h._slack_team_id = None
        h.command = "POST"
        for k, v in attrs.items():
            setattr(h, k, v)
        return h

    return _make


def _make_agent(name: str, elo: float = 1500, wins: int = 0, losses: int = 0):
    """Create a mock agent object with ELO attributes."""
    return SimpleNamespace(name=name, elo=elo, wins=wins, losses=losses)


# ---------------------------------------------------------------------------
# _command_help
# ---------------------------------------------------------------------------


class TestCommandHelp:
    """Tests for the help subcommand."""

    def test_help_returns_usage_text(self, slack_handler):
        result = slack_handler._command_help()
        body = _body(result)
        assert "response_type" in body
        assert body["response_type"] == "ephemeral"
        assert "Aragora Slash Commands" in body["text"]

    def test_help_mentions_all_commands(self, slack_handler):
        body = _body(slack_handler._command_help())
        for cmd in ("debate", "plan", "implement", "ask", "gauntlet",
                     "search", "recent", "leaderboard", "agents", "status", "help"):
            assert cmd in body["text"], f"Help text should mention '{cmd}'"

    def test_help_includes_examples(self, slack_handler):
        body = _body(slack_handler._command_help())
        assert "Examples:" in body["text"]


# ---------------------------------------------------------------------------
# _command_status
# ---------------------------------------------------------------------------


class TestCommandStatus:
    """Tests for the status subcommand."""

    @patch("aragora.server.handlers.social._slack_impl.commands.EloSystem", create=True)
    def test_status_returns_blocks(self, mock_elo_cls, slack_handler):
        """Status returns blocks with agent count when ELO system available."""
        with patch.dict("sys.modules", {}):
            mock_store = MagicMock()
            mock_store.get_all_ratings.return_value = [_make_agent("a1"), _make_agent("a2")]
            mock_elo_cls.return_value = mock_store

            # Patch the import inside the method
            with patch(
                "aragora.ranking.elo.EloSystem", mock_elo_cls, create=True
            ):
                result = slack_handler._command_status()
                body = _body(result)
                assert _status(result) == 200
                assert "blocks" in body

    def test_status_import_error(self, slack_handler):
        """Status returns fallback when ELO system not importable."""
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "unavailable" in body["text"].lower() or "error" in body["text"].lower()

    def test_status_data_error(self, slack_handler):
        """Status handles data errors gracefully."""
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = TypeError("bad data")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "error" in body["text"].lower()

    def test_status_connection_error(self, slack_handler):
        """Status handles connection errors gracefully."""
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = ConnectionError("down")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "error" in body["text"].lower()


# ---------------------------------------------------------------------------
# _command_agents
# ---------------------------------------------------------------------------


class TestCommandAgents:
    """Tests for the agents subcommand."""

    def test_agents_empty(self, slack_handler):
        """Empty agent list returns informational message."""
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = []
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_agents()
            body = _body(result)
            assert "no agents" in body["text"].lower()

    def test_agents_sorted_by_elo(self, slack_handler):
        """Agents listed in descending ELO order."""
        agents = [
            _make_agent("low", elo=1200, wins=2),
            _make_agent("high", elo=1800, wins=10),
            _make_agent("mid", elo=1500, wins=5),
        ]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_agents()
            body = _body(result)
            text = body["text"]
            # high should appear before mid, mid before low
            assert text.index("high") < text.index("mid") < text.index("low")

    def test_agents_limits_to_10(self, slack_handler):
        """Only top 10 agents shown."""
        agents = [_make_agent(f"agent{i}", elo=2000 - i * 10) for i in range(15)]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_agents()
            body = _body(result)
            # agent10 through agent14 should NOT appear
            assert "agent14" not in body["text"]

    def test_agents_import_error(self, slack_handler):
        """Agents handles import error gracefully."""
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = slack_handler._command_agents()
            body = _body(result)
            assert "unavailable" in body["text"].lower()

    def test_agents_attribute_error(self, slack_handler):
        """Agents handles attribute errors gracefully."""
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = AttributeError("no attr")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_agents()
            body = _body(result)
            assert "error" in body["text"].lower()


# ---------------------------------------------------------------------------
# _command_ask
# ---------------------------------------------------------------------------


class TestCommandAsk:
    """Tests for the ask subcommand."""

    def test_ask_no_args(self, slack_handler):
        result = slack_handler._command_ask("", "U1", "C1", "https://hooks.slack.com/resp")
        body = _body(result)
        assert "provide a question" in body["text"].lower()

    def test_ask_too_short(self, slack_handler):
        result = slack_handler._command_ask("Hi", "U1", "C1", "https://hooks.slack.com/resp")
        body = _body(result)
        assert "too short" in body["text"].lower()

    def test_ask_too_long(self, slack_handler):
        long_q = "x" * 501
        result = slack_handler._command_ask(long_q, "U1", "C1", "https://hooks.slack.com/resp")
        body = _body(result)
        assert "too long" in body["text"].lower()

    def test_ask_strips_quotes(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_ask(
            '"What is the meaning of life?"', "U1", "C1", "https://hooks.slack.com/resp"
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert "blocks" in body

    def test_ask_queues_async_task(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_ask(
            "What is quantum computing used for?",
            "U1",
            "C1",
            "https://hooks.slack.com/resp",
        )
        mock_create.assert_called_once()

    def test_ask_no_response_url_skips_task(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_ask(
            "What is quantum computing used for?", "U1", "C1", ""
        )
        mock_create.assert_not_called()

    def test_ask_response_includes_user_id(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_ask(
            "Explain artificial intelligence in depth", "UABC", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        # User should be mentioned in context blocks
        found = False
        for b in body.get("blocks", []):
            for elem in b.get("elements", []):
                if "UABC" in elem.get("text", ""):
                    found = True
        assert found, "User ID should appear in response blocks"

    def test_ask_exactly_5_chars(self, slack_handler, commands_module, monkeypatch):
        """Edge case: 5-char question should be accepted."""
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_ask("Hello", "U1", "C1", "https://hooks.slack.com/x")
        body = _body(result)
        # Should NOT be rejected as too short
        assert "too short" not in body.get("text", "").lower()

    def test_ask_exactly_500_chars(self, slack_handler, commands_module, monkeypatch):
        """Edge case: 500-char question should be accepted."""
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        q = "x" * 500
        result = slack_handler._command_ask(q, "U1", "C1", "https://hooks.slack.com/x")
        body = _body(result)
        assert "too long" not in body.get("text", "").lower()


# ---------------------------------------------------------------------------
# _command_search
# ---------------------------------------------------------------------------


class TestCommandSearch:
    """Tests for the search subcommand."""

    def test_search_no_args(self, slack_handler):
        result = slack_handler._command_search("")
        body = _body(result)
        assert "provide a search query" in body["text"].lower()

    def test_search_too_short(self, slack_handler):
        result = slack_handler._command_search("a")
        body = _body(result)
        assert "too short" in body["text"].lower()

    def test_search_with_db_search_method(self, slack_handler, commands_module, monkeypatch):
        """Search uses db.search() when available."""
        mock_db = MagicMock()
        mock_db.search.return_value = (
            [{"task": "AI topic", "consensus_reached": True, "id": "abc123", "confidence": 0.85}],
            1,
        )
        mock_get_db = MagicMock(return_value=mock_db)
        monkeypatch.setattr(commands_module, "get_debates_db", mock_get_db)

        result = slack_handler._command_search("AI topic")
        body = _body(result)
        assert "blocks" in body
        assert "1 result" in body["text"]

    def test_search_with_db_list_fallback(self, slack_handler, commands_module, monkeypatch):
        """Search uses db.list() fallback when db.search() not available."""
        mock_db = MagicMock(spec=["list"])
        mock_db.list.return_value = [
            {"task": "Machine learning basics", "final_answer": "ML is...", "id": "xyz",
             "consensus_reached": False, "confidence": 0.5},
        ]
        mock_get_db = MagicMock(return_value=mock_db)
        monkeypatch.setattr(commands_module, "get_debates_db", mock_get_db)

        result = slack_handler._command_search("machine")
        body = _body(result)
        assert "blocks" in body

    def test_search_no_results(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.search.return_value = ([], 0)
        mock_get_db = MagicMock(return_value=mock_db)
        monkeypatch.setattr(commands_module, "get_debates_db", mock_get_db)

        result = slack_handler._command_search("nonexistent")
        body = _body(result)
        assert "no results" in body["text"].lower()

    def test_search_db_unavailable(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "get_debates_db", None)
        result = slack_handler._command_search("some query")
        body = _body(result)
        assert "failed" in body["text"].lower() or "unavailable" in body["text"].lower()

    def test_search_db_returns_none(self, slack_handler, commands_module, monkeypatch):
        mock_get_db = MagicMock(return_value=None)
        monkeypatch.setattr(commands_module, "get_debates_db", mock_get_db)
        result = slack_handler._command_search("some query")
        body = _body(result)
        assert "no results" in body["text"].lower()

    def test_search_object_format_results(self, slack_handler, commands_module, monkeypatch):
        """Search handles object-format results (not dicts)."""
        item = SimpleNamespace(
            task="Object topic", consensus_reached=True, id="obj123", confidence=0.9
        )
        mock_db = MagicMock()
        mock_db.search.return_value = ([item], 1)
        mock_get_db = MagicMock(return_value=mock_db)
        monkeypatch.setattr(commands_module, "get_debates_db", mock_get_db)

        result = slack_handler._command_search("object")
        body = _body(result)
        assert "blocks" in body

    def test_search_limits_to_5_results(self, slack_handler, commands_module, monkeypatch):
        items = [
            {"task": f"Topic {i}", "consensus_reached": True, "id": f"id{i}", "confidence": 0.5}
            for i in range(10)
        ]
        mock_db = MagicMock()
        mock_db.search.return_value = (items, 10)
        mock_get_db = MagicMock(return_value=mock_db)
        monkeypatch.setattr(commands_module, "get_debates_db", mock_get_db)

        result = slack_handler._command_search("Topic")
        body = _body(result)
        # Should only show 5 result blocks (plus header and context)
        section_blocks = [b for b in body.get("blocks", []) if b.get("type") == "section"]
        assert len(section_blocks) <= 5

    def test_search_strips_quotes(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.search.return_value = ([], 0)
        mock_get_db = MagicMock(return_value=mock_db)
        monkeypatch.setattr(commands_module, "get_debates_db", mock_get_db)

        slack_handler._command_search('"quoted query"')
        mock_db.search.assert_called_once_with("quoted query", limit=5)


# ---------------------------------------------------------------------------
# _command_leaderboard
# ---------------------------------------------------------------------------


class TestCommandLeaderboard:
    """Tests for the leaderboard subcommand."""

    def test_leaderboard_empty(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = []
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_leaderboard()
            body = _body(result)
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
            result = slack_handler._command_leaderboard()
            body = _body(result)
            # Context block should mention total agents
            found = False
            for b in body.get("blocks", []):
                for elem in b.get("elements", []):
                    if "5" in elem.get("text", ""):
                        found = True
            assert found

    def test_leaderboard_import_error(self, slack_handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = slack_handler._command_leaderboard()
            body = _body(result)
            assert "unavailable" in body["text"].lower()

    def test_leaderboard_attribute_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = AttributeError("bad")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_leaderboard()
            body = _body(result)
            assert "failed" in body["text"].lower()


# ---------------------------------------------------------------------------
# _command_recent
# ---------------------------------------------------------------------------


class TestCommandRecent:
    """Tests for the recent subcommand."""

    def test_recent_db_unavailable(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "get_debates_db", None)
        result = slack_handler._command_recent()
        body = _body(result)
        assert "failed" in body["text"].lower() or "unavailable" in body["text"].lower()

    def test_recent_db_returns_none(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=None))
        result = slack_handler._command_recent()
        body = _body(result)
        assert "not available" in body["text"].lower()

    def test_recent_db_no_list_method(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock(spec=[])  # no list attr
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_recent()
        body = _body(result)
        assert "not available" in body["text"].lower()

    def test_recent_empty(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.list.return_value = []
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_recent()
        body = _body(result)
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
            task="Object debate",
            consensus_reached=False,
            confidence=0.4,
            id="obj-d1",
            created_at="2026-02-01T12:00:00",
        )
        mock_db = MagicMock()
        mock_db.list.return_value = [obj]
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_recent()
        body = _body(result)
        assert "blocks" in body

    def test_recent_includes_details_button(self, slack_handler, commands_module, monkeypatch):
        debates = [
            {"task": "Button test", "consensus_reached": True, "confidence": 0.8,
             "id": "btn123", "created_at": "2026-01-01"},
        ]
        mock_db = MagicMock()
        mock_db.list.return_value = debates
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_recent()
        body = _body(result)
        # Find a section block with an accessory button
        found_button = False
        for b in body.get("blocks", []):
            acc = b.get("accessory", {})
            if acc.get("action_id") == "view_details":
                found_button = True
                assert acc["value"] == "btn123"
        assert found_button

    def test_recent_truncates_long_topic(self, slack_handler, commands_module, monkeypatch):
        debates = [
            {"task": "A" * 100, "consensus_reached": True, "confidence": 0.5,
             "id": "trunc1", "created_at": "2026-01-01"},
        ]
        mock_db = MagicMock()
        mock_db.list.return_value = debates
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_recent()
        body = _body(result)
        # Should have ... for truncated topic
        blocks_text = json.dumps(body.get("blocks", []))
        assert "..." in blocks_text

    def test_recent_runtime_error(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.list.side_effect = RuntimeError("db down")
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_recent()
        body = _body(result)
        assert "failed" in body["text"].lower()


# ---------------------------------------------------------------------------
# _command_debate
# ---------------------------------------------------------------------------


class TestCommandDebate:
    """Tests for the debate subcommand."""

    def test_debate_no_args(self, slack_handler):
        result = slack_handler._command_debate(
            "", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "provide a topic" in body["text"].lower()

    def test_debate_topic_too_short(self, slack_handler):
        result = slack_handler._command_debate(
            "short", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too short" in body["text"].lower()

    def test_debate_topic_too_long(self, slack_handler):
        result = slack_handler._command_debate(
            "x" * 501, "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too long" in body["text"].lower()

    def test_debate_valid_topic(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        result = slack_handler._command_debate(
            "Should AI be regulated by governments?",
            "U1",
            "C1",
            "https://hooks.slack.com/x",
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert "blocks" in body

    def test_debate_queues_async_task_with_response_url(
        self, slack_handler, commands_module, monkeypatch
    ):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_debate(
            "Should AI be regulated by governments?",
            "U1",
            "C1",
            "https://hooks.slack.com/x",
        )
        mock_create.assert_called_once()

    def test_debate_no_response_url_skips_task(
        self, slack_handler, commands_module, monkeypatch
    ):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_debate(
            "Should AI be regulated by governments?",
            "U1",
            "C1",
            "",
        )
        mock_create.assert_not_called()

    def test_debate_strips_quotes(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_debate(
            '"Should we adopt microservices architecture?"',
            "U1", "C1", "https://hooks.slack.com/x",
        )
        body = _body(result)
        assert "microservices" in json.dumps(body.get("blocks", []))

    def test_debate_plan_mode_label(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_debate(
            "Should we refactor the backend API?",
            "U1", "C1", "https://hooks.slack.com/x",
            mode_label="plan",
        )
        body = _body(result)
        blocks_text = json.dumps(body.get("blocks", []))
        assert "plan" in blocks_text

    def test_debate_no_args_with_command_label(self, slack_handler):
        result = slack_handler._command_debate(
            "", "U1", "C1", "https://hooks.slack.com/x",
            command_label="implement",
        )
        body = _body(result)
        assert "implement" in body["text"].lower()

    def test_debate_exactly_10_chars(self, slack_handler, commands_module, monkeypatch):
        """Edge case: 10-char topic should be accepted."""
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_debate(
            "0123456789", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too short" not in body.get("text", "").lower()


# ---------------------------------------------------------------------------
# _command_gauntlet
# ---------------------------------------------------------------------------


class TestCommandGauntlet:
    """Tests for the gauntlet subcommand."""

    def test_gauntlet_no_args(self, slack_handler):
        result = slack_handler._command_gauntlet(
            "", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "provide a statement" in body["text"].lower()

    def test_gauntlet_too_short(self, slack_handler):
        result = slack_handler._command_gauntlet(
            "short", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too short" in body["text"].lower()

    def test_gauntlet_too_long(self, slack_handler):
        result = slack_handler._command_gauntlet(
            "x" * 1001, "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too long" in body["text"].lower()

    def test_gauntlet_valid_statement(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        result = slack_handler._command_gauntlet(
            "We should migrate to microservices architecture",
            "U1",
            "C1",
            "https://hooks.slack.com/x",
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

    def test_gauntlet_no_response_url_skips_task(
        self, slack_handler, commands_module, monkeypatch
    ):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_gauntlet(
            "We should migrate to microservices architecture",
            "U1", "C1", "",
        )
        mock_create.assert_not_called()

    def test_gauntlet_strips_quotes(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_gauntlet(
            '"We should adopt a monorepo strategy for our codebase"',
            "U1", "C1", "https://hooks.slack.com/x",
        )
        body = _body(result)
        blocks_text = json.dumps(body.get("blocks", []))
        assert "monorepo" in blocks_text

    def test_gauntlet_exactly_10_chars(self, slack_handler, commands_module, monkeypatch):
        """Edge case: exactly 10 chars accepted."""
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_gauntlet(
            "0123456789", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too short" not in body.get("text", "").lower()

    def test_gauntlet_exactly_1000_chars(self, slack_handler, commands_module, monkeypatch):
        """Edge case: exactly 1000 chars accepted."""
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_gauntlet(
            "x" * 1000, "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too long" not in body.get("text", "").lower()


# ---------------------------------------------------------------------------
# _handle_slash_command dispatch
# ---------------------------------------------------------------------------


class TestHandleSlashCommand:
    """Tests for the top-level slash command dispatcher."""

    def _make_handler_with_body(self, text: str = "", command: str = "/aragora",
                                 user_id: str = "U123", channel_id: str = "C456",
                                 team_id: str = "T789"):
        h = MagicMock()
        form = {
            "command": command,
            "text": text,
            "user_id": user_id,
            "channel_id": channel_id,
            "response_url": "https://hooks.slack.com/resp/123",
        }
        if team_id:
            form["team_id"] = team_id
        h._slack_body = urlencode(form)
        h._slack_workspace = None
        h._slack_team_id = team_id
        return h

    def test_dispatch_help_explicit(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = self._make_handler_with_body("help")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "Aragora Slash Commands" in body["text"]

    def test_dispatch_help_no_text(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = self._make_handler_with_body("")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "Aragora Slash Commands" in body["text"]

    def test_dispatch_unknown_command(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = self._make_handler_with_body("foobar")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "unknown command" in body["text"].lower()

    def test_dispatch_status(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        # Mock status to avoid ELO import
        slack_handler._command_status = MagicMock(
            return_value=slack_handler._slack_response("Status OK")
        )
        h = self._make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        slack_handler._command_status.assert_called_once()

    def test_dispatch_debate(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = self._make_handler_with_body('debate "Should AI be regulated?"')
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_dispatch_plan(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = self._make_handler_with_body('plan "Refactor the entire backend codebase"')
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_dispatch_implement(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = self._make_handler_with_body('implement "Build a new authentication system"')
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_dispatch_agents(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_agents = MagicMock(
            return_value=slack_handler._slack_response("Agents list")
        )
        h = self._make_handler_with_body("agents")
        result = slack_handler._handle_slash_command(h)
        slack_handler._command_agents.assert_called_once()

    def test_dispatch_ask(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = self._make_handler_with_body('ask "What is the capital of France?"')
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_dispatch_search(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_search = MagicMock(
            return_value=slack_handler._slack_response("Results")
        )
        h = self._make_handler_with_body("search machine learning")
        result = slack_handler._handle_slash_command(h)
        slack_handler._command_search.assert_called_once_with("machine learning")

    def test_dispatch_leaderboard(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_leaderboard = MagicMock(
            return_value=slack_handler._slack_response("Leaderboard")
        )
        h = self._make_handler_with_body("leaderboard")
        result = slack_handler._handle_slash_command(h)
        slack_handler._command_leaderboard.assert_called_once()

    def test_dispatch_recent(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_recent = MagicMock(
            return_value=slack_handler._slack_response("Recent")
        )
        h = self._make_handler_with_body("recent")
        result = slack_handler._handle_slash_command(h)
        slack_handler._command_recent.assert_called_once()

    def test_dispatch_gauntlet(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = self._make_handler_with_body('gauntlet "We should adopt microservices"')
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_dispatch_case_insensitive(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = self._make_handler_with_body("HELP")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "Aragora Slash Commands" in body["text"]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for workspace and user rate limiting in _handle_slash_command."""

    def _make_handler_with_body(self, text="help", team_id="T789"):
        h = MagicMock()
        form = {
            "command": "/aragora",
            "text": text,
            "user_id": "U123",
            "channel_id": "C456",
            "response_url": "https://hooks.slack.com/resp/123",
        }
        if team_id:
            form["team_id"] = team_id
        h._slack_body = urlencode(form)
        h._slack_workspace = None
        h._slack_team_id = team_id
        return h

    def test_workspace_rate_limited(self, slack_handler, commands_module, monkeypatch):
        """Workspace rate limit blocks the command."""
        ws_limiter = MagicMock()
        ws_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=30)
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)

        h = self._make_handler_with_body()
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "too quickly" in body["text"].lower()
        assert "30" in body["text"]

    def test_user_rate_limited(self, slack_handler, commands_module, monkeypatch):
        """User rate limit blocks the command."""
        user_limiter = MagicMock()
        user_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=15)
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: user_limiter)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)

        h = self._make_handler_with_body()
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "too quickly" in body["text"].lower()
        assert "15" in body["text"]

    def test_workspace_rate_limit_audit_logged(self, slack_handler, commands_module, monkeypatch):
        """Workspace rate limit triggers audit log."""
        ws_limiter = MagicMock()
        ws_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=10)
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)

        h = self._make_handler_with_body()
        slack_handler._handle_slash_command(h)
        mock_audit.log_rate_limit.assert_called_once()
        call_kwargs = mock_audit.log_rate_limit.call_args.kwargs
        assert call_kwargs["limit_type"] == "workspace"

    def test_user_rate_limit_audit_logged(self, slack_handler, commands_module, monkeypatch):
        """User rate limit triggers audit log."""
        user_limiter = MagicMock()
        user_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=5)
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: user_limiter)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)

        h = self._make_handler_with_body()
        slack_handler._handle_slash_command(h)
        mock_audit.log_rate_limit.assert_called_once()
        call_kwargs = mock_audit.log_rate_limit.call_args.kwargs
        assert call_kwargs["limit_type"] == "user"

    def test_both_limiters_allow_passes_through(self, slack_handler, commands_module, monkeypatch):
        """When both rate limiters allow, command proceeds normally."""
        ws_limiter = MagicMock()
        ws_limiter.allow.return_value = SimpleNamespace(allowed=True, retry_after=0)
        user_limiter = MagicMock()
        user_limiter.allow.return_value = SimpleNamespace(allowed=True, retry_after=0)
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: user_limiter)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)

        h = self._make_handler_with_body("help")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "Aragora Slash Commands" in body["text"]

    def test_no_team_id_skips_workspace_rate_limit(
        self, slack_handler, commands_module, monkeypatch
    ):
        """When team_id is None, workspace rate limiting is skipped."""
        ws_limiter = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)

        h = self._make_handler_with_body("help", team_id="")
        h._slack_team_id = None
        result = slack_handler._handle_slash_command(h)
        ws_limiter.allow.assert_not_called()


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


class TestAuditLogging:
    """Tests for audit logging on command success and error."""

    def _make_handler_with_body(self, text="help"):
        h = MagicMock()
        form = {
            "command": "/aragora",
            "text": text,
            "user_id": "U123",
            "channel_id": "C456",
            "response_url": "https://hooks.slack.com/resp",
        }
        h._slack_body = urlencode(form)
        h._slack_workspace = None
        h._slack_team_id = "T789"
        return h

    def test_audit_logged_on_success(self, slack_handler, commands_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)

        h = self._make_handler_with_body("help")
        slack_handler._handle_slash_command(h)
        mock_audit.log_command.assert_called_once()
        call_kwargs = mock_audit.log_command.call_args.kwargs
        assert call_kwargs["result"] == "success"

    def test_audit_logged_on_error(self, slack_handler, commands_module, monkeypatch):
        """When a subcommand raises ValueError, audit logs error."""
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)

        # Make status raise ValueError
        slack_handler._command_status = MagicMock(side_effect=ValueError("test error"))
        h = self._make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "error" in body["text"].lower()
        # Should have been called with result="error"
        assert mock_audit.log_command.call_count >= 1
        error_calls = [
            c for c in mock_audit.log_command.call_args_list
            if c.kwargs.get("result") == "error"
        ]
        assert len(error_calls) == 1

    def test_no_audit_when_logger_none(self, slack_handler, commands_module, monkeypatch):
        """No error when audit logger is None."""
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)

        h = self._make_handler_with_body("help")
        result = slack_handler._handle_slash_command(h)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# _answer_question_async
# ---------------------------------------------------------------------------


class TestAnswerQuestionAsync:
    """Tests for the async question answering flow."""

    @pytest.mark.asyncio
    async def test_answer_question_happy_path(self, slack_handler):
        """Successful quick-answer API call posts answer to response_url."""
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"answer": "Paris is the capital of France"}
        mock_session.post.return_value = mock_response

        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._answer_question_async(
                "What is the capital of France?",
                "https://hooks.slack.com/resp",
                "U1",
                "C1",
            )
            slack_handler._post_to_response_url.assert_called_once()
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "Paris" in json.dumps(payload.get("blocks", []))

    @pytest.mark.asyncio
    async def test_answer_question_fallback_to_debate(self, slack_handler):
        """Falls back to debate API when quick-answer returns no answer."""
        mock_session = AsyncMock()

        # First call: quick-answer returns no answer
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"answer": None}
        # Second call: debate API returns answer
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"final_answer": "The debate concluded that..."}

        mock_session.post.side_effect = [resp1, resp2]

        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._answer_question_async(
                "Explain deep learning", "https://hooks.slack.com/resp", "U1", "C1"
            )
            slack_handler._post_to_response_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_question_error_posts_failure(self, slack_handler):
        """Network errors post failure message."""
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
            slack_handler._post_to_response_url.assert_called_once()
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "failed" in payload["text"].lower()


# ---------------------------------------------------------------------------
# _run_gauntlet_async
# ---------------------------------------------------------------------------


class TestRunGauntletAsync:
    """Tests for the async gauntlet execution flow."""

    @pytest.mark.asyncio
    async def test_gauntlet_async_happy_path(self, slack_handler):
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

        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Test statement for validation",
                "https://hooks.slack.com/resp",
                "U1",
                "C1",
            )
            slack_handler._post_to_response_url.assert_called_once()
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "blocks" in payload

    @pytest.mark.asyncio
    async def test_gauntlet_async_with_vulnerabilities(self, slack_handler):
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

        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Weak statement here!",
                "https://hooks.slack.com/resp",
                "U1",
                "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            blocks_text = json.dumps(payload.get("blocks", []))
            assert "Lacks evidence" in blocks_text

    @pytest.mark.asyncio
    async def test_gauntlet_async_api_error(self, slack_handler):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_session.post.return_value = mock_response

        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Some statement here",
                "https://hooks.slack.com/resp",
                "U1",
                "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "failed" in payload["text"].lower()

    @pytest.mark.asyncio
    async def test_gauntlet_async_network_error(self, slack_handler):
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
                "https://hooks.slack.com/resp",
                "U1",
                "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "failed" in payload["text"].lower()


# ---------------------------------------------------------------------------
# _update_debate_status
# ---------------------------------------------------------------------------


class TestUpdateDebateStatus:
    """Tests for _update_debate_status."""

    def test_update_debate_status_store_available(self, slack_handler):
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

    def test_update_debate_status_store_unavailable(self, slack_handler):
        """No error when slack debate store is not importable."""
        with patch.dict("sys.modules", {"aragora.storage.slack_debate_store": None}):
            # Should not raise
            slack_handler._update_debate_status("d1", "failed", error="some error")

    def test_update_debate_status_with_error(self, slack_handler):
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


# ---------------------------------------------------------------------------
# _create_debate_async
# ---------------------------------------------------------------------------


class TestCreateDebateAsync:
    """Tests for the async debate creation flow."""

    @pytest.mark.asyncio
    async def test_create_debate_async_posts_starting_message(self, slack_handler, commands_module, monkeypatch):
        """Starting message is posted to response_url when no bot token."""
        monkeypatch.setattr(commands_module, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        # Also patch in blocks module since it imports SLACK_BOT_TOKEN
        try:
            from aragora.server.handlers.social._slack_impl import blocks as blocks_mod
            monkeypatch.setattr(blocks_mod, "SLACK_BOT_TOKEN", None)
        except (ImportError, AttributeError):
            pass

        # Mock imports inside method
        mock_env = MagicMock()
        mock_protocol = MagicMock()
        mock_protocol.rounds = 3
        mock_arena = MagicMock()
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.rounds_used = 3
        mock_result.participants = ["anthropic-api", "openai-api"]
        mock_result.final_answer = "The agents concluded that..."
        mock_result.winner = None
        mock_result.id = "test-debate-id"
        mock_arena.run = AsyncMock(return_value=mock_result)

        slack_handler._post_to_response_url = AsyncMock()
        slack_handler._update_debate_status = MagicMock()

        with patch("aragora.server.handlers.social._slack_impl.commands.register_debate_origin", create=True):
            with patch("aragora.Environment", mock_env, create=True):
                with patch("aragora.DebateProtocol", return_value=mock_protocol, create=True):
                    with patch("aragora.Arena", create=True) as mock_arena_cls:
                        mock_arena_cls.from_env.return_value = mock_arena
                        with patch("aragora.agents.get_agents_by_names", return_value=["a1", "a2"], create=True):
                            with patch("aragora.server.handlers.social._slack_impl.commands.maybe_emit_decision_integrity", new_callable=AsyncMock, create=True):
                                await slack_handler._create_debate_async(
                                    "Should AI be regulated?",
                                    "https://hooks.slack.com/resp",
                                    "U1",
                                    "C1",
                                    "WS1",
                                )

        # Should have posted at least starting + result
        assert slack_handler._post_to_response_url.call_count >= 1


# ---------------------------------------------------------------------------
# Error handling in _handle_slash_command
# ---------------------------------------------------------------------------


class TestSlashCommandErrorHandling:
    """Tests for error handling in the top-level slash command dispatcher."""

    def _make_handler_with_body(self, text="status"):
        h = MagicMock()
        form = {
            "command": "/aragora",
            "text": text,
            "user_id": "U123",
            "channel_id": "C456",
            "response_url": "https://hooks.slack.com/resp",
            "team_id": "T789",
        }
        h._slack_body = urlencode(form)
        h._slack_workspace = None
        h._slack_team_id = "T789"
        return h

    def test_value_error_caught(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_status = MagicMock(side_effect=ValueError("bad"))
        h = self._make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "error" in body["text"].lower()

    def test_key_error_caught(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_status = MagicMock(side_effect=KeyError("missing"))
        h = self._make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "error" in body["text"].lower()

    def test_type_error_caught(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_status = MagicMock(side_effect=TypeError("wrong type"))
        h = self._make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "error" in body["text"].lower()

    def test_runtime_error_caught(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_status = MagicMock(side_effect=RuntimeError("crash"))
        h = self._make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "error" in body["text"].lower()

    def test_os_error_caught(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_status = MagicMock(side_effect=OSError("disk"))
        h = self._make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "error" in body["text"].lower()

    def test_connection_error_caught(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        slack_handler._command_status = MagicMock(side_effect=ConnectionError("down"))
        h = self._make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "error" in body["text"].lower()

    def test_error_audit_includes_response_time(self, slack_handler, commands_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        slack_handler._command_status = MagicMock(side_effect=ValueError("fail"))
        h = self._make_handler_with_body("status")
        slack_handler._handle_slash_command(h)
        call_kwargs = mock_audit.log_command.call_args.kwargs
        assert "response_time_ms" in call_kwargs
        assert call_kwargs["response_time_ms"] >= 0


# ---------------------------------------------------------------------------
# Plan and implement modes
# ---------------------------------------------------------------------------


class TestPlanAndImplementModes:
    """Tests for plan and implement subcommands via the dispatcher."""

    def _make_handler_with_body(self, text):
        h = MagicMock()
        form = {
            "command": "/aragora",
            "text": text,
            "user_id": "U123",
            "channel_id": "C456",
            "response_url": "https://hooks.slack.com/resp",
            "team_id": "T789",
        }
        h._slack_body = urlencode(form)
        h._slack_workspace = None
        h._slack_team_id = "T789"
        return h

    def test_plan_passes_decision_integrity(self, slack_handler, commands_module, monkeypatch):
        """Plan mode passes decision_integrity config to _command_debate."""
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        captured = {}
        original_debate = slack_handler._command_debate

        def spy_debate(*args, **kwargs):
            captured.update(kwargs)
            return original_debate(*args, **kwargs)

        slack_handler._command_debate = spy_debate
        h = self._make_handler_with_body('plan "Refactor the backend to be much better"')
        slack_handler._handle_slash_command(h)
        assert captured.get("decision_integrity") is not None
        assert captured["decision_integrity"]["include_plan"] is True
        assert captured["decision_integrity"]["include_context"] is False
        assert captured.get("mode_label") == "plan"

    def test_implement_passes_execution_mode(self, slack_handler, commands_module, monkeypatch):
        """Implement mode passes execution_mode in decision_integrity."""
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        captured = {}
        original_debate = slack_handler._command_debate

        def spy_debate(*args, **kwargs):
            captured.update(kwargs)
            return original_debate(*args, **kwargs)

        slack_handler._command_debate = spy_debate
        h = self._make_handler_with_body('implement "Build new auth system for the app"')
        slack_handler._handle_slash_command(h)
        di = captured.get("decision_integrity", {})
        assert di.get("execution_mode") == "execute"
        assert di.get("include_context") is True
        assert captured.get("mode_label") == "implementation plan"
        assert captured.get("command_label") == "implement"


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_slash_command_with_extra_whitespace(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
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
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "Aragora Slash Commands" in body["text"]

    def test_search_list_fallback_no_match(self, slack_handler, commands_module, monkeypatch):
        """List fallback with no matching results returns no results message."""
        mock_db = MagicMock(spec=["list"])
        mock_db.list.return_value = [
            {"task": "Unrelated topic", "final_answer": "Something else",
             "id": "x", "consensus_reached": False, "confidence": 0},
        ]
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_search("quantum")
        body = _body(result)
        assert "no results" in body["text"].lower()

    def test_ask_4_chars_rejected(self, slack_handler):
        """4-char question is too short."""
        result = slack_handler._command_ask("Hiya", "U1", "C1", "https://hooks.slack.com/x")
        body = _body(result)
        assert "too short" in body["text"].lower()

    def test_gauntlet_9_chars_rejected(self, slack_handler):
        """9-char statement is too short for gauntlet."""
        result = slack_handler._command_gauntlet(
            "123456789", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too short" in body["text"].lower()

    def test_debate_9_chars_rejected(self, slack_handler):
        """9-char topic is too short for debate."""
        result = slack_handler._command_debate(
            "123456789", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert "too short" in body["text"].lower()

    def test_all_responses_are_ephemeral_or_in_channel(self, slack_handler, commands_module, monkeypatch):
        """All responses should have a valid response_type."""
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        # Help
        body = _body(slack_handler._command_help())
        assert body["response_type"] in ("ephemeral", "in_channel")

    def test_search_connection_error(self, slack_handler, commands_module, monkeypatch):
        """Search handles ValueError from DB."""
        mock_db = MagicMock()
        mock_db.search.side_effect = ValueError("bad query")
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_search("valid query")
        body = _body(result)
        assert "failed" in body["text"].lower()

    def test_agents_medals_for_top_3(self, slack_handler):
        """Top 3 agents get medal emojis."""
        agents = [
            _make_agent("first", elo=2000, wins=20),
            _make_agent("second", elo=1900, wins=15),
            _make_agent("third", elo=1800, wins=10),
            _make_agent("fourth", elo=1700, wins=5),
        ]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_agents()
            body = _body(result)
            # Medals are unicode chars
            assert "\U0001f947" in body["text"]  # gold
            assert "\U0001f948" in body["text"]  # silver
            assert "\U0001f949" in body["text"]  # bronze
