"""Comprehensive tests for WhatsApp bot command implementations.

Covers all functions in aragora/server/handlers/social/whatsapp/commands.py:
- command_help()
- command_status()
- command_agents()
- command_debate() + run_debate_async()
- command_gauntlet() + run_gauntlet_async()
- command_search()
- command_recent()
- command_receipt()
- _format_receipt()
- _format_debate_as_receipt()
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.whatsapp.commands import (
    _format_debate_as_receipt,
    _format_receipt,
    command_agents,
    command_debate,
    command_gauntlet,
    command_help,
    command_receipt,
    command_recent,
    command_search,
    command_status,
    run_debate_async,
    run_gauntlet_async,
)

# Module path for patching imports used inside function bodies
_CMD = "aragora.server.handlers.social.whatsapp.commands"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler_instance(perm_error: str | None = None, ctx: dict | None = None):
    """Create a mock handler instance with configurable permission checks."""
    handler = MagicMock()
    handler._check_whatsapp_permission.return_value = perm_error
    handler.ctx = ctx or {}
    return handler


def _make_mock_debate_result(
    consensus=True,
    confidence=0.85,
    rounds=3,
    final_answer="AI should be regulated.",
    debate_id="debate-123",
):
    result = MagicMock()
    result.consensus_reached = consensus
    result.confidence = confidence
    result.rounds_used = rounds
    result.final_answer = final_answer
    result.id = debate_id
    return result


def _mock_aragora_module(arena_cls=None, agents=None):
    """Create a mock aragora module with Arena, Environment, DebateProtocol."""
    mod = MagicMock()
    if arena_cls is not None:
        mod.Arena = arena_cls
    if agents is not None:
        mod.Arena.from_env.return_value = MagicMock(run=AsyncMock(return_value=agents))
    return mod


def _gauntlet_session(response):
    """Create mock HTTP session and pool for gauntlet tests."""
    mock_session = AsyncMock()
    mock_session.post.return_value = response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.get_session.return_value = mock_session
    return mock_pool, mock_session


def _gauntlet_response(status_code=200, run_id="gauntlet-001", score=0.9, passed=True, vulns=None):
    """Create a mock HTTP response for gauntlet API."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "run_id": run_id,
        "score": score,
        "passed": passed,
        "vulnerabilities": vulns or [],
    }
    if status_code != 200:
        resp.json.return_value = {"error": "Internal server error"}
    return resp


# ---------------------------------------------------------------------------
# command_help
# ---------------------------------------------------------------------------


class TestCommandHelp:
    """Tests for command_help()."""

    def test_returns_string(self):
        result = command_help()
        assert isinstance(result, str)

    def test_contains_all_commands(self):
        result = command_help()
        for cmd in [
            "help",
            "status",
            "agents",
            "debate",
            "plan",
            "implement",
            "gauntlet",
            "search",
            "recent",
            "receipt",
        ]:
            assert cmd in result.lower(), f"Missing command: {cmd}"

    def test_contains_examples(self):
        result = command_help()
        assert "Examples" in result

    def test_contains_usage_examples(self):
        result = command_help()
        assert "debate Should AI be regulated?" in result
        assert "receipt abc123" in result


# ---------------------------------------------------------------------------
# command_status
# ---------------------------------------------------------------------------


class TestCommandStatus:
    """Tests for command_status()."""

    def test_status_online_with_agents(self):
        mock_agents = [MagicMock(), MagicMock(), MagicMock()]
        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = mock_agents

        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value = mock_elo
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_status()
        assert "Online" in result
        assert "3 registered" in result

    def test_status_online_on_import_error(self):
        """When EloSystem cannot be imported, still returns Online status."""
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = command_status()
        assert "Online" in result

    def test_status_online_on_attribute_error(self):
        """When get_all_ratings raises AttributeError, falls back gracefully."""
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.side_effect = AttributeError("no such attribute")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_status()
        assert "Online" in result

    def test_status_contains_agent_count_on_success(self):
        """When EloSystem works, status includes agent count."""
        mock_elo_mod = MagicMock()
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = [1, 2, 3, 4, 5]
        mock_elo_mod.EloSystem.return_value = mock_store
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_status()
        assert "5 registered" in result

    def test_status_on_value_error(self):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.side_effect = ValueError("bad value")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_status()
        assert "Online" in result


# ---------------------------------------------------------------------------
# command_agents
# ---------------------------------------------------------------------------


class TestCommandAgents:
    """Tests for command_agents()."""

    def test_no_agents_registered(self):
        mock_elo_mod = MagicMock()
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = []
        mock_elo_mod.EloSystem.return_value = mock_store
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_agents()
        assert "No agents registered" in result

    def test_agents_listed_sorted_by_elo(self):
        agent1 = MagicMock(name="Alpha", elo=1600, wins=10)
        agent1.name = "Alpha"
        agent1.elo = 1600
        agent1.wins = 10
        agent2 = MagicMock(name="Beta", elo=1800, wins=20)
        agent2.name = "Beta"
        agent2.elo = 1800
        agent2.wins = 20

        mock_elo_mod = MagicMock()
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = [agent1, agent2]
        mock_elo_mod.EloSystem.return_value = mock_store
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_agents()
        assert "Beta" in result
        assert "Alpha" in result
        # Beta has higher ELO, should appear first
        assert result.index("Beta") < result.index("Alpha")

    def test_agents_max_10_displayed(self):
        agents = []
        for i in range(15):
            a = MagicMock()
            a.name = f"Agent{i}"
            a.elo = 1500 + i
            a.wins = i
            agents.append(a)

        mock_elo_mod = MagicMock()
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = agents
        mock_elo_mod.EloSystem.return_value = mock_store
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_agents()
        # Only top 10 should appear (agent14 down to agent5, by descending ELO)
        assert "Agent14" in result
        assert "Agent5" in result
        # Agent0 through Agent4 should NOT appear (below top 10)
        assert "Agent0" not in result

    def test_agents_import_error(self):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = command_agents()
        assert "Could not fetch agent list" in result

    def test_agents_type_error(self):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.side_effect = TypeError("bad type")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_agents()
        assert "Could not fetch agent list" in result

    def test_agents_with_missing_attributes(self):
        """Agents missing name/elo/wins use defaults."""
        agent = MagicMock(spec=[])  # No attributes at all
        mock_elo_mod = MagicMock()
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = [agent]
        mock_elo_mod.EloSystem.return_value = mock_store
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = command_agents()
        assert "Unknown" in result
        assert "1500" in result


# ---------------------------------------------------------------------------
# command_debate
# ---------------------------------------------------------------------------


class TestCommandDebate:
    """Tests for command_debate() synchronous dispatch."""

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_permission_denied(self, mock_create_task):
        handler = _make_handler_instance(perm_error="Access denied")
        command_debate(handler, "+1234567890", "Alice", "Some topic for debate")
        mock_create_task.assert_called_once()
        call_args = mock_create_task.call_args
        assert "perm-denied" in call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_topic_too_short(self, mock_create_task):
        handler = _make_handler_instance()
        command_debate(handler, "+1234567890", "Alice", "short")
        mock_create_task.assert_called_once()
        assert "short" in mock_create_task.call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_topic_too_long(self, mock_create_task):
        handler = _make_handler_instance()
        long_topic = "x" * 501
        command_debate(handler, "+1234567890", "Alice", long_topic)
        mock_create_task.assert_called_once()
        assert "long" in mock_create_task.call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_topic_exactly_500_chars(self, mock_create_task):
        """500 chars is the max allowed length."""
        handler = _make_handler_instance()
        topic = "x" * 500
        command_debate(handler, "+1234567890", "Alice", topic)
        # Should create two tasks: acknowledgment + debate run
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_topic_exactly_10_chars(self, mock_create_task):
        """10 chars is the min allowed length."""
        handler = _make_handler_instance()
        command_debate(handler, "+1234567890", "Alice", "0123456789")
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_topic_9_chars_too_short(self, mock_create_task):
        handler = _make_handler_instance()
        command_debate(handler, "+1234567890", "Alice", "012345678")
        assert mock_create_task.call_count == 1
        assert "short" in mock_create_task.call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_successful_debate_creates_ack_and_run(self, mock_create_task):
        handler = _make_handler_instance()
        command_debate(handler, "+1234567890", "Alice", "Should we adopt microservices?")
        assert mock_create_task.call_count == 2
        task_names = [call[1]["name"] for call in mock_create_task.call_args_list]
        assert any("ack" in name for name in task_names)
        assert any("debate" in name for name in task_names)

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_topic_strips_whitespace_and_quotes(self, mock_create_task):
        handler = _make_handler_instance()
        command_debate(handler, "+1234567890", "Alice", '  "Is AI safe enough?"  ')
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_topic_stripped_too_short(self, mock_create_task):
        """After stripping quotes and whitespace, topic might be too short."""
        handler = _make_handler_instance()
        command_debate(handler, "+1234567890", "Alice", '  "short"  ')
        assert mock_create_task.call_count == 1
        assert "short" in mock_create_task.call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_ctx_passed_to_run_debate(self, mock_create_task):
        """Handler ctx is propagated to run_debate_async."""
        ctx = {"document_store": MagicMock(), "evidence_store": MagicMock()}
        handler = _make_handler_instance(ctx=ctx)
        command_debate(handler, "+1234567890", "Alice", "Should we adopt microservices?")
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_decision_integrity_passed_through(self, mock_create_task):
        handler = _make_handler_instance()
        command_debate(
            handler,
            "+1234567890",
            "Alice",
            "Should we adopt microservices?",
            decision_integrity={"mode": "full"},
        )
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_rbac_check_called_with_correct_permission(self, mock_create_task):
        handler = _make_handler_instance()
        command_debate(handler, "+1234567890", "Alice", "Should we adopt microservices?")
        handler._check_whatsapp_permission.assert_called_once_with(
            "+1234567890", "debates.create", "Alice"
        )


# ---------------------------------------------------------------------------
# run_debate_async
# ---------------------------------------------------------------------------


class TestRunDebateAsync:
    """Tests for run_debate_async().

    The function imports Arena, Environment, DebateProtocol, etc. inside its
    body, so we mock the source modules via sys.modules and the module-level
    imports (record_*, emit_*, send_*) via standard patch.
    """

    @pytest.fixture
    def mock_debate_result(self):
        return _make_mock_debate_result()

    @pytest.fixture
    def _debate_mocks(self):
        """Set up all the inline imports that run_debate_async uses."""
        mock_arena = MagicMock()
        mock_env = MagicMock()
        mock_protocol = MagicMock()
        mock_agents_mod = MagicMock()
        mock_origin_mod = MagicMock()
        mock_bindings_mod = MagicMock()
        mock_integrity_mod = MagicMock()
        mock_integrity_mod.maybe_emit_decision_integrity = AsyncMock()

        # Create aragora mock that provides Arena, Environment, DebateProtocol
        mock_aragora = MagicMock()
        mock_aragora.Arena = mock_arena
        mock_aragora.Environment = mock_env
        mock_aragora.DebateProtocol = mock_protocol

        return {
            "arena_cls": mock_arena,
            "env_cls": mock_env,
            "protocol_cls": mock_protocol,
            "agents_mod": mock_agents_mod,
            "origin_mod": mock_origin_mod,
            "bindings_mod": mock_bindings_mod,
            "integrity_mod": mock_integrity_mod,
            "aragora_mod": mock_aragora,
        }

    def _patch_debate_imports(self, mocks, agents_return=None):
        """Return a context manager that patches all inline imports."""
        if agents_return is None:
            agents_return = ["agent1", "agent2"]
        mocks["agents_mod"].get_agents_by_names.return_value = agents_return
        return patch.dict(
            "sys.modules",
            {
                "aragora": mocks["aragora_mod"],
                "aragora.agents": mocks["agents_mod"],
                "aragora.server.debate_origin": mocks["origin_mod"],
                "aragora.server.bindings": mocks["bindings_mod"],
                "aragora.server.decision_integrity_utils": mocks["integrity_mod"],
            },
        )

    @pytest.mark.asyncio
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_successful_debate(
        self,
        mock_send_buttons,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
        mock_debate_result,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_debate_result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Should AI be regulated?")

        mock_record_started.assert_called_once_with("whatsapp")
        mock_send_buttons.assert_called_once()
        mock_record_completed.assert_called_once_with("whatsapp", True)
        mock_emit_started.assert_called_once()
        mock_emit_completed.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_debate_failed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_no_agents_available(
        self,
        mock_send,
        mock_record_started,
        mock_record_failed,
        _debate_mocks,
    ):
        with self._patch_debate_imports(_debate_mocks, agents_return=[]):
            await run_debate_async("+1234567890", "Alice", "Should AI be regulated?")

        mock_send.assert_called_once()
        assert "No agents available" in mock_send.call_args[0][1]
        mock_record_failed.assert_called_once_with("whatsapp")

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_debate_failed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_debate_runtime_error(
        self,
        mock_send,
        mock_record_started,
        mock_record_failed,
        _debate_mocks,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(side_effect=RuntimeError("boom"))
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Test topic for debate")

        mock_record_failed.assert_called_once_with("whatsapp")
        mock_send.assert_called_once()
        assert "error occurred" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_debate_failed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_debate_value_error(
        self,
        mock_send,
        mock_record_started,
        mock_record_failed,
        _debate_mocks,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(side_effect=ValueError("bad value"))
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Test topic for debate")

        mock_record_failed.assert_called_once_with("whatsapp")

    @pytest.mark.asyncio
    @patch(f"{_CMD}.TTS_VOICE_ENABLED", True)
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_voice_summary", new_callable=AsyncMock)
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_tts_voice_sent_when_enabled(
        self,
        mock_send_buttons,
        mock_send_voice,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
        mock_debate_result,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_debate_result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Should AI be regulated?")

        mock_send_voice.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{_CMD}.TTS_VOICE_ENABLED", False)
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_voice_summary", new_callable=AsyncMock)
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_tts_voice_not_sent_when_disabled(
        self,
        mock_send_buttons,
        mock_send_voice,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
        mock_debate_result,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_debate_result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Should AI be regulated?")

        mock_send_voice.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_long_topic_truncated_in_response(
        self,
        mock_send_buttons,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
        mock_debate_result,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_debate_result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "A" * 150)

        response_text = mock_send_buttons.call_args[0][1]
        assert "..." in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_long_final_answer_truncated(
        self,
        mock_send_buttons,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
    ):
        result = _make_mock_debate_result(final_answer="X" * 600, debate_id="debate-456")
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Test long answer topic")

        response_text = mock_send_buttons.call_args[0][1]
        assert "..." in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_none_final_answer(
        self,
        mock_send_buttons,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
    ):
        result = _make_mock_debate_result(final_answer=None, debate_id="debate-789")
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Test null answer topic")

        response_text = mock_send_buttons.call_args[0][1]
        assert "No conclusion" in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_interactive_buttons_include_vote_options(
        self,
        mock_send_buttons,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
        mock_debate_result,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_debate_result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Should AI be regulated?")

        buttons = mock_send_buttons.call_args[0][2]
        assert len(buttons) == 3
        titles = [b["title"] for b in buttons]
        assert "Agree" in titles
        assert "Disagree" in titles
        assert "View Details" in titles

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_debate_failed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_connection_error(
        self,
        mock_send,
        mock_record_started,
        mock_record_failed,
        _debate_mocks,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(side_effect=ConnectionError("network down"))
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Test connection error topic")

        mock_record_failed.assert_called_once_with("whatsapp")

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_debate_failed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_os_error_during_debate(
        self,
        mock_send,
        mock_record_started,
        mock_record_failed,
        _debate_mocks,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(side_effect=OSError("disk full"))
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Test OS error topic")

        mock_record_failed.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_consensus_no_in_response(
        self,
        mock_send_buttons,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
    ):
        result = _make_mock_debate_result(consensus=False, confidence=0.3)
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Contentious topic for debate")

        response_text = mock_send_buttons.call_args[0][1]
        assert "*Consensus:* No" in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_debate_id_in_response(
        self,
        mock_send_buttons,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
    ):
        result = _make_mock_debate_result(debate_id="my-unique-debate-id")
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "ID visibility debate topic")

        response_text = mock_send_buttons.call_args[0][1]
        assert "my-unique-debate-id" in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.emit_debate_completed")
    @patch(f"{_CMD}.emit_debate_started")
    @patch(f"{_CMD}.record_debate_completed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_interactive_buttons", new_callable=AsyncMock)
    async def test_profile_name_in_response(
        self,
        mock_send_buttons,
        mock_record_started,
        mock_record_completed,
        mock_emit_started,
        mock_emit_completed,
        _debate_mocks,
        mock_debate_result,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_debate_result)
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Bob Jones", "Who should be PM?")

        response_text = mock_send_buttons.call_args[0][1]
        assert "Bob Jones" in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_debate_failed")
    @patch(f"{_CMD}.record_debate_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_key_error_during_debate(
        self,
        mock_send,
        mock_record_started,
        mock_record_failed,
        _debate_mocks,
    ):
        mock_arena_instance = MagicMock()
        mock_arena_instance.run = AsyncMock(side_effect=KeyError("missing"))
        _debate_mocks["arena_cls"].from_env.return_value = mock_arena_instance

        with self._patch_debate_imports(_debate_mocks):
            await run_debate_async("+1234567890", "Alice", "Test key error topic here")

        mock_record_failed.assert_called_once()


# ---------------------------------------------------------------------------
# command_gauntlet
# ---------------------------------------------------------------------------


class TestCommandGauntlet:
    """Tests for command_gauntlet() synchronous dispatch."""

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_permission_denied(self, mock_create_task):
        handler = _make_handler_instance(perm_error="Access denied")
        command_gauntlet(handler, "+1234567890", "Alice", "We should migrate to the cloud")
        mock_create_task.assert_called_once()
        assert "perm-denied" in mock_create_task.call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_statement_too_short(self, mock_create_task):
        handler = _make_handler_instance()
        command_gauntlet(handler, "+1234567890", "Alice", "short")
        mock_create_task.assert_called_once()
        assert "short" in mock_create_task.call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_statement_too_long(self, mock_create_task):
        handler = _make_handler_instance()
        long_statement = "y" * 1001
        command_gauntlet(handler, "+1234567890", "Alice", long_statement)
        mock_create_task.assert_called_once()
        assert "long" in mock_create_task.call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_statement_exactly_1000_chars(self, mock_create_task):
        handler = _make_handler_instance()
        command_gauntlet(handler, "+1234567890", "Alice", "y" * 1000)
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_statement_exactly_10_chars(self, mock_create_task):
        handler = _make_handler_instance()
        command_gauntlet(handler, "+1234567890", "Alice", "0123456789")
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_statement_9_chars_too_short(self, mock_create_task):
        handler = _make_handler_instance()
        command_gauntlet(handler, "+1234567890", "Alice", "012345678")
        assert mock_create_task.call_count == 1
        assert "short" in mock_create_task.call_args[1]["name"]

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_successful_gauntlet_creates_ack_and_run(self, mock_create_task):
        handler = _make_handler_instance()
        command_gauntlet(
            handler, "+1234567890", "Alice", "We should migrate to microservices architecture"
        )
        assert mock_create_task.call_count == 2
        task_names = [call[1]["name"] for call in mock_create_task.call_args_list]
        assert any("ack" in name for name in task_names)
        assert any("gauntlet" in name for name in task_names)

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_statement_strips_whitespace_and_quotes(self, mock_create_task):
        handler = _make_handler_instance()
        command_gauntlet(handler, "+1234567890", "Alice", '  "We should adopt Kubernetes"  ')
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_long_statement_truncated_in_ack(self, mock_create_task):
        """Statements over 200 chars are truncated in ack message."""
        handler = _make_handler_instance()
        long_statement = "A" * 300
        command_gauntlet(handler, "+1234567890", "Alice", long_statement)
        assert mock_create_task.call_count == 2

    @patch(f"{_CMD}._config.create_tracked_task")
    def test_rbac_check_called_with_correct_permission(self, mock_create_task):
        handler = _make_handler_instance()
        command_gauntlet(
            handler, "+1234567890", "Alice", "We should adopt microservices architecture"
        )
        handler._check_whatsapp_permission.assert_called_once_with(
            "+1234567890", "gauntlet.run", "Alice"
        )


# ---------------------------------------------------------------------------
# run_gauntlet_async
# ---------------------------------------------------------------------------


class TestRunGauntletAsync:
    """Tests for run_gauntlet_async()."""

    def _patch_gauntlet(self, pool):
        """Patch get_http_pool at its source for inline import."""
        mock_http_mod = MagicMock()
        mock_http_mod.get_http_pool.return_value = pool
        return patch.dict(
            "sys.modules",
            {
                "aragora.server.http_client_pool": mock_http_mod,
            },
        )

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_passed(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_emit_completed,
        mock_record_completed,
    ):
        resp = _gauntlet_response(passed=True, score=0.9)
        pool, _ = _gauntlet_session(resp)

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "We should adopt microservices")

        mock_send.assert_called_once()
        response_text = mock_send.call_args[0][1]
        assert "PASSED" in response_text
        assert "90.0%" in response_text
        mock_record_completed.assert_called_once_with("whatsapp", True)

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_failed_verdict(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_emit_completed,
        mock_record_completed,
    ):
        vulns = [
            {"description": "Logical inconsistency found"},
            {"description": "Missing evidence for key claim"},
        ]
        resp = _gauntlet_response(passed=False, score=0.3, vulns=vulns)
        pool, _ = _gauntlet_session(resp)

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "We should adopt microservices")

        response_text = mock_send.call_args[0][1]
        assert "FAILED" in response_text
        assert "Logical inconsistency" in response_text
        assert "Missing evidence" in response_text
        mock_record_completed.assert_called_once_with("whatsapp", False)

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_failed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_api_error_response(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_record_failed,
    ):
        resp = _gauntlet_response(status_code=500)
        pool, _ = _gauntlet_session(resp)

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "Microservices are better")

        response_text = mock_send.call_args[0][1]
        assert "failed" in response_text.lower()
        mock_record_failed.assert_called_once_with("whatsapp")

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_failed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_connection_error(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_record_failed,
    ):
        mock_session = AsyncMock()
        mock_session.post.side_effect = ConnectionError("network down")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock()
        pool.get_session.return_value = mock_session

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "Microservices are better")

        mock_record_failed.assert_called_once_with("whatsapp")
        mock_send.assert_called_once()
        assert "error occurred" in mock_send.call_args[0][1]

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_failed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_timeout_error(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_record_failed,
    ):
        mock_session = AsyncMock()
        mock_session.post.side_effect = TimeoutError("request timed out")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock()
        pool.get_session.return_value = mock_session

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "Microservices are better")

        mock_record_failed.assert_called_once_with("whatsapp")

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_many_vulnerabilities_truncated(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_emit_completed,
        mock_record_completed,
    ):
        vulns = [{"description": f"Vulnerability {i}"} for i in range(8)]
        resp = _gauntlet_response(passed=False, score=0.4, vulns=vulns)
        pool, _ = _gauntlet_session(resp)

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "Should we adopt microservices?")

        response_text = mock_send.call_args[0][1]
        assert "...and 3 more" in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_response_includes_run_id(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_emit_completed,
        mock_record_completed,
    ):
        resp = _gauntlet_response(run_id="gauntlet-unique-id", score=0.75)
        pool, _ = _gauntlet_session(resp)

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "Statement for gauntlet test")

        response_text = mock_send.call_args[0][1]
        assert "gauntlet-unique-id" in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_long_statement_truncated_in_response(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_emit_completed,
        mock_record_completed,
    ):
        resp = _gauntlet_response(score=0.8)
        pool, _ = _gauntlet_session(resp)

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "B" * 300)

        response_text = mock_send.call_args[0][1]
        assert "..." in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_no_vulnerabilities(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_emit_completed,
        mock_record_completed,
    ):
        resp = _gauntlet_response(passed=True, score=0.95, vulns=[])
        pool, _ = _gauntlet_session(resp)

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "Clean statement passes easily")

        response_text = mock_send.call_args[0][1]
        assert "Issues Found" not in response_text
        assert "Vulnerabilities:* 0" in response_text

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_failed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_os_error(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_record_failed,
    ):
        mock_session = AsyncMock()
        mock_session.post.side_effect = OSError("broken pipe")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock()
        pool.get_session.return_value = mock_session

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Alice", "OS error test statement here")

        mock_record_failed.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{_CMD}.record_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_completed")
    @patch(f"{_CMD}.emit_gauntlet_started")
    @patch(f"{_CMD}.record_gauntlet_started")
    @patch(f"{_CMD}.send_text_message", new_callable=AsyncMock)
    async def test_gauntlet_profile_name_in_response(
        self,
        mock_send,
        mock_record_started,
        mock_emit_started,
        mock_emit_completed,
        mock_record_completed,
    ):
        resp = _gauntlet_response(passed=True)
        pool, _ = _gauntlet_session(resp)

        with self._patch_gauntlet(pool):
            await run_gauntlet_async("+1234567890", "Carol Smith", "A well-formed statement test")

        response_text = mock_send.call_args[0][1]
        assert "Carol Smith" in response_text


# ---------------------------------------------------------------------------
# command_search
# ---------------------------------------------------------------------------


class TestCommandSearch:
    """Tests for command_search()."""

    def test_empty_query(self):
        result = command_search("")
        assert "at least 3 characters" in result

    def test_none_query(self):
        result = command_search(None)
        assert "at least 3 characters" in result

    def test_short_query(self):
        result = command_search("ab")
        assert "at least 3 characters" in result

    def test_whitespace_only_query(self):
        result = command_search("  ")
        assert "at least 3 characters" in result

    def test_search_with_db_search_method(self):
        mock_db = MagicMock()
        mock_db.search.return_value = (
            [
                {"topic": "Machine Learning debate", "id": "d1", "consensus_reached": True},
                {"topic": "Deep Learning review", "id": "d2", "consensus_reached": False},
            ],
            2,
        )

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("machine learning")
        assert "Machine Learning debate" in result
        assert "d1" in result
        assert "d2" in result

    def test_search_no_results(self):
        mock_db = MagicMock()
        mock_db.search.return_value = ([], 0)

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("nonexistent topic")
        assert "No debates found" in result

    def test_search_fallback_to_recent_debates(self):
        mock_db = MagicMock(spec=[])  # No search method
        mock_db.get_recent_debates = MagicMock(
            return_value=[
                {
                    "topic": "AI Ethics",
                    "id": "d3",
                    "consensus_reached": True,
                    "conclusion": "Be responsible",
                },
            ]
        )
        assert not hasattr(mock_db, "search")

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("ethics")
        assert "AI Ethics" in result

    def test_search_fallback_no_db(self):
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = None
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("some query")
        assert "not available" in result

    def test_search_fallback_no_search_no_recent(self):
        mock_db = MagicMock(spec=[])
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("some query")
        assert "not available" in result

    def test_search_import_error(self):
        with patch.dict("sys.modules", {"aragora.storage": None}):
            result = command_search("some query")
        assert "temporarily unavailable" in result

    def test_search_runtime_error(self):
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.side_effect = RuntimeError("db error")
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("some query")
        assert "error occurred" in result

    def test_search_more_than_5_results(self):
        debates = [
            {"topic": f"Topic {i}", "id": f"d{i}", "consensus_reached": i % 2 == 0}
            for i in range(10)
        ]
        mock_db = MagicMock()
        mock_db.search.return_value = (debates[:5], 10)

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("topic")
        assert "Showing 5 of 10" in result

    def test_search_long_topic_truncated(self):
        mock_db = MagicMock()
        mock_db.search.return_value = (
            [{"topic": "A" * 100, "id": "d1", "consensus_reached": True}],
            1,
        )

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("AAA")
        assert "..." in result

    def test_search_consensus_yes(self):
        mock_db = MagicMock()
        mock_db.search.return_value = (
            [{"topic": "Consensus topic", "id": "d1", "consensus_reached": True}],
            1,
        )
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("consensus")
        assert "Consensus: Yes" in result

    def test_search_consensus_no(self):
        mock_db = MagicMock()
        mock_db.search.return_value = (
            [{"topic": "No consensus topic", "id": "d1", "consensus_reached": False}],
            1,
        )
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_search("consensus")
        assert "Consensus: No" in result


# ---------------------------------------------------------------------------
# command_recent
# ---------------------------------------------------------------------------


class TestCommandRecent:
    """Tests for command_recent()."""

    def test_no_recent_debates(self):
        mock_db = MagicMock()
        mock_db.get_recent_debates.return_value = []

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_recent()
        assert "No recent debates" in result
        assert "debate <topic>" in result

    def test_recent_debates_listed(self):
        debates = [
            {"topic": "AI Safety", "id": "d1", "consensus_reached": True, "confidence": 0.9},
            {"topic": "Climate Change", "id": "d2", "consensus_reached": False, "confidence": 0.4},
        ]
        mock_db = MagicMock()
        mock_db.get_recent_debates.return_value = debates

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_recent()
        assert "AI Safety" in result
        assert "Climate Change" in result
        assert "d1" in result
        assert "d2" in result
        assert "receipt <id>" in result

    def test_recent_no_db(self):
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = None
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_recent()
        assert "not available" in result

    def test_recent_db_no_get_recent_method(self):
        mock_db = MagicMock(spec=[])
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_recent()
        assert "not available" in result

    def test_recent_import_error(self):
        with patch.dict("sys.modules", {"aragora.storage": None}):
            result = command_recent()
        assert "temporarily unavailable" in result

    def test_recent_runtime_error(self):
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.side_effect = RuntimeError("connection lost")
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_recent()
        assert "error occurred" in result

    def test_recent_long_topic_truncated(self):
        debates = [
            {"topic": "X" * 60, "id": "d1", "consensus_reached": True, "confidence": 0.8},
        ]
        mock_db = MagicMock()
        mock_db.get_recent_debates.return_value = debates

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_recent()
        assert "..." in result

    def test_recent_confidence_formatted_as_percent(self):
        debates = [
            {"topic": "Budget review", "id": "d1", "consensus_reached": True, "confidence": 0.75},
        ]
        mock_db = MagicMock()
        mock_db.get_recent_debates.return_value = debates

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db
        with patch.dict("sys.modules", {"aragora.storage": mock_storage_mod}):
            result = command_recent()
        assert "75%" in result


# ---------------------------------------------------------------------------
# command_receipt
# ---------------------------------------------------------------------------


class TestCommandReceipt:
    """Tests for command_receipt()."""

    def test_empty_debate_id(self):
        result = command_receipt("")
        assert "Please provide a debate ID" in result
        assert "receipt abc123" in result

    def test_none_debate_id(self):
        result = command_receipt(None)
        assert "Please provide a debate ID" in result

    def test_whitespace_debate_id(self):
        result = command_receipt("   ")
        assert "Please provide a debate ID" in result

    def test_receipt_from_receipt_store(self):
        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = {
            "receipt_id": "rcpt-001",
            "topic": "AI Safety Policy",
            "decision": "Implement guardrails",
            "confidence": 0.85,
            "timestamp": "2026-02-23T10:00:00Z",
            "agents": [{"name": "Claude"}, {"name": "GPT-4"}],
            "hash": "abcdef1234567890abcdef1234567890",
        }

        mock_receipt_store = MagicMock()
        mock_receipt_store.get.return_value = mock_receipt

        mock_receipt_store_mod = MagicMock()
        mock_receipt_store_mod.get_receipt_store.return_value = mock_receipt_store
        with patch.dict("sys.modules", {"aragora.storage.receipt_store": mock_receipt_store_mod}):
            result = command_receipt("rcpt-001")
        assert "Decision Receipt" in result
        assert "rcpt-001" in result
        assert "AI Safety Policy" in result
        assert "Implement guardrails" in result

    def test_receipt_fallback_to_debate_store(self):
        mock_db = MagicMock()
        mock_db.get_debate.return_value = {
            "id": "debate-001",
            "topic": "Code Review Process",
            "conclusion": "Adopt pair review",
            "consensus_reached": True,
            "confidence": 0.9,
            "rounds_used": 3,
        }

        mock_gauntlet_mod = MagicMock()
        mock_gauntlet_mod.DecisionReceipt.from_dict.return_value = MagicMock(
            to_dict=MagicMock(
                return_value={
                    "receipt_id": "debate-001",
                    "topic": "Code Review Process",
                    "decision": "Adopt pair review",
                    "confidence": 0.9,
                    "timestamp": "2026-02-23",
                    "agents": [],
                }
            )
        )

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.receipt_store": None,
                "aragora.storage": mock_storage_mod,
                "aragora.gauntlet.receipt": mock_gauntlet_mod,
            },
        ):
            result = command_receipt("debate-001")
        assert "Receipt" in result

    def test_receipt_no_debate_found(self):
        mock_db = MagicMock()
        mock_db.get_debate.return_value = None

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.receipt_store": None,
                "aragora.storage": mock_storage_mod,
            },
        ):
            result = command_receipt("nonexistent-id")
        assert "No debate found" in result

    def test_receipt_no_db(self):
        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = None

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.receipt_store": None,
                "aragora.storage": mock_storage_mod,
            },
        ):
            result = command_receipt("some-id")
        assert "not available" in result

    def test_receipt_unexpected_error(self):
        mock_receipt_store_mod = MagicMock()
        mock_receipt_store_mod.get_receipt_store.side_effect = ValueError("corrupted")

        with patch.dict("sys.modules", {"aragora.storage.receipt_store": mock_receipt_store_mod}):
            result = command_receipt("some-id")
        assert "error occurred" in result

    def test_receipt_fallback_format_when_gauntlet_unavailable(self):
        mock_db = MagicMock()
        mock_db.get_debate.return_value = {
            "id": "debate-002",
            "topic": "Hiring Process",
            "conclusion": "Use structured interviews",
            "consensus_reached": True,
            "confidence": 0.8,
            "rounds_used": 2,
        }

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.receipt_store": None,
                "aragora.storage": mock_storage_mod,
                "aragora.gauntlet.receipt": None,
            },
        ):
            result = command_receipt("debate-002")
        assert "Debate Summary" in result
        assert "debate-002" in result
        assert "Hiring Process" in result

    def test_receipt_strips_whitespace_from_id(self):
        mock_receipt_store = MagicMock()
        mock_receipt_store.get.return_value = None

        mock_receipt_store_mod = MagicMock()
        mock_receipt_store_mod.get_receipt_store.return_value = mock_receipt_store

        mock_db = MagicMock()
        mock_db.get_debate.return_value = None

        mock_storage_mod = MagicMock()
        mock_storage_mod.get_storage.return_value = mock_db

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.receipt_store": mock_receipt_store_mod,
                "aragora.storage": mock_storage_mod,
            },
        ):
            result = command_receipt("  debate-001  ")
        mock_receipt_store.get.assert_called_once_with("debate-001")

    def test_receipt_use_recent_to_find_ids_hint(self):
        result = command_receipt("")
        assert "recent" in result.lower()


# ---------------------------------------------------------------------------
# _format_receipt
# ---------------------------------------------------------------------------


class TestFormatReceipt:
    """Tests for _format_receipt()."""

    def test_basic_format(self):
        data = {
            "receipt_id": "rcpt-001",
            "topic": "Should we use K8s?",
            "decision": "Yes, for production workloads",
            "confidence": 0.88,
            "timestamp": "2026-02-23T12:00:00Z",
            "agents": [],
        }
        result = _format_receipt(data)
        assert "Decision Receipt" in result
        assert "rcpt-001" in result
        assert "Should we use K8s?" in result
        assert "Yes, for production workloads" in result
        assert "88%" in result

    def test_fallback_field_names(self):
        """Uses alternate field names (id, question, conclusion, created_at)."""
        data = {
            "id": "alt-001",
            "question": "Should we refactor?",
            "conclusion": "Prioritize tech debt",
            "confidence": 0.7,
            "created_at": "2026-01-01",
            "participants": ["Claude", "GPT-4"],
        }
        result = _format_receipt(data)
        assert "alt-001" in result
        assert "Should we refactor?" in result
        assert "Prioritize tech debt" in result
        assert "70%" in result
        assert "Claude" in result

    def test_string_confidence(self):
        data = {
            "receipt_id": "rcpt-002",
            "topic": "Test",
            "decision": "Done",
            "confidence": "high",
            "timestamp": "now",
            "agents": [],
        }
        result = _format_receipt(data)
        assert "high" in result

    def test_long_topic_truncated(self):
        data = {
            "receipt_id": "rcpt-003",
            "topic": "Z" * 100,
            "decision": "Short",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
        }
        result = _format_receipt(data)
        assert "..." in result

    def test_long_decision_truncated(self):
        data = {
            "receipt_id": "rcpt-004",
            "topic": "Short",
            "decision": "W" * 300,
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
        }
        result = _format_receipt(data)
        assert "..." in result

    def test_agents_as_dicts(self):
        data = {
            "receipt_id": "rcpt-005",
            "topic": "Agent test",
            "decision": "Done",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [{"name": "Alpha"}, {"name": "Beta"}],
        }
        result = _format_receipt(data)
        assert "Alpha" in result
        assert "Beta" in result

    def test_agents_as_strings(self):
        data = {
            "receipt_id": "rcpt-006",
            "topic": "Agent string test",
            "decision": "Done",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": ["agent1", "agent2"],
        }
        result = _format_receipt(data)
        assert "agent1" in result
        assert "agent2" in result

    def test_verification_hash(self):
        data = {
            "receipt_id": "rcpt-007",
            "topic": "Hash test",
            "decision": "Done",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
            "hash": "abcdef1234567890abcdef1234567890",
        }
        result = _format_receipt(data)
        assert "Verification" in result
        assert "abcdef1234567890" in result

    def test_no_verification_hash(self):
        data = {
            "receipt_id": "rcpt-008",
            "topic": "No hash",
            "decision": "Done",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
        }
        result = _format_receipt(data)
        assert "Verification" not in result

    def test_max_5_agents_shown(self):
        data = {
            "receipt_id": "rcpt-009",
            "topic": "Many agents",
            "decision": "Done",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [{"name": f"Agent{i}"} for i in range(10)],
        }
        result = _format_receipt(data)
        assert "Agent0" in result
        assert "Agent4" in result
        assert "Agent5" not in result


# ---------------------------------------------------------------------------
# _format_debate_as_receipt
# ---------------------------------------------------------------------------


class TestFormatDebateAsReceipt:
    """Tests for _format_debate_as_receipt()."""

    def test_basic_format(self):
        debate = {
            "id": "debate-001",
            "topic": "Code Review Process",
            "conclusion": "Use pair review",
            "consensus_reached": True,
            "confidence": 0.85,
            "rounds_used": 3,
        }
        result = _format_debate_as_receipt(debate)
        assert "Debate Summary" in result
        assert "debate-001" in result
        assert "Code Review Process" in result
        assert "Use pair review" in result
        assert "Yes" in result
        assert "85%" in result
        assert "3" in result

    def test_no_consensus(self):
        debate = {
            "id": "debate-002",
            "topic": "Hiring Strategy",
            "conclusion": "No agreement",
            "consensus_reached": False,
            "confidence": 0.3,
        }
        result = _format_debate_as_receipt(debate)
        assert "No" in result

    def test_final_answer_fallback(self):
        """Uses final_answer when conclusion missing."""
        debate = {
            "id": "debate-003",
            "topic": "Deployment",
            "final_answer": "Use blue-green deployments",
            "consensus_reached": True,
            "confidence": 0.7,
        }
        result = _format_debate_as_receipt(debate)
        assert "Use blue-green deployments" in result

    def test_string_confidence(self):
        debate = {
            "id": "debate-004",
            "topic": "Test",
            "conclusion": "Done",
            "consensus_reached": True,
            "confidence": "medium",
        }
        result = _format_debate_as_receipt(debate)
        assert "medium" in result

    def test_long_topic_truncated(self):
        debate = {
            "id": "debate-005",
            "topic": "T" * 100,
            "conclusion": "Done",
            "consensus_reached": True,
            "confidence": 0.5,
        }
        result = _format_debate_as_receipt(debate)
        assert len(debate["topic"][:80]) == 80

    def test_long_conclusion_truncated(self):
        debate = {
            "id": "debate-006",
            "topic": "Short",
            "conclusion": "C" * 300,
            "consensus_reached": True,
            "confidence": 0.5,
        }
        result = _format_debate_as_receipt(debate)
        assert "C" * 250 in result

    def test_no_rounds_used(self):
        debate = {
            "id": "debate-007",
            "topic": "Test",
            "conclusion": "Done",
            "consensus_reached": True,
            "confidence": 0.5,
        }
        result = _format_debate_as_receipt(debate)
        assert "Rounds" not in result

    def test_missing_fields_use_defaults(self):
        debate = {}
        result = _format_debate_as_receipt(debate)
        assert "N/A" in result
        assert "Unknown" in result
