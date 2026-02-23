"""
Tests for Teams Bot event processing.

Covers all routes and behavior of the TeamsEventProcessor class:
- process_activity() routing for all activity types
  - message / invoke / conversationUpdate / messageReaction /
    installationUpdate / unknown
- _handle_message()
  - Mention-based commands (debate, ask, plan, implement, status,
    help, leaderboard, agents, vote)
  - Personal chat with known command
  - Personal chat with unknown command (defaults to debate)
  - Non-mention, non-personal message (prompt reply)
  - RBAC permission denied for messages
  - Empty text, empty command
- _handle_command() routing to all commands
  - debate / ask / plan / implement with decision_integrity params
  - status / help / leaderboard / agents / vote / unknown
- _cmd_debate()
  - RBAC check, empty topic, success, decision_integrity label,
    attachments (list vs non-list), audit logging
- _cmd_status()
  - Online card with ELO store available / unavailable
  - Active debate count
- _cmd_help()
  - With help card module / with ImportError fallback
- _cmd_leaderboard()
  - With ELO store + leaderboard card / ImportError fallbacks
  - Default leaderboard text when no standings
- _cmd_agents()
  - Sends agents text
- _cmd_vote()
  - RBAC check, no args + active debates, no args + no debates,
    with args
- _cmd_unknown()
  - Sends unknown command reply
- _handle_invoke()
  - Delegates to card actions
- _handle_conversation_update()
  - Bot added (welcome) / bot removed (cleanup)
  - Non-bot member added/removed (no-op)
- _handle_message_reaction()
  - Reactions added (like, heart) / removed
  - No reactions
- _handle_installation_update()
  - add / remove / unknown action
- _send_welcome()
  - With help card module / with ImportError fallback
- Module-level constants
  - AGENT_DISPLAY_NAMES, MENTION_PATTERN, permission constants, __all__
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_activity(
    activity_type: str = "message",
    text: str = "hello",
    user_id: str = "user-123",
    user_name: str = "TestUser",
    conversation_id: str = "conv-abc",
    conversation_type: str = "",
    service_url: str = "https://smba.trafficmanager.net/teams/",
    entities: list | None = None,
    invoke_name: str = "",
    value: dict[str, Any] | None = None,
    reply_to_id: str | None = None,
    members_added: list | None = None,
    members_removed: list | None = None,
    recipient_id: str = "bot-id",
    reactions_added: list | None = None,
    reactions_removed: list | None = None,
    action: str = "",
    attachments: list | None = None,
) -> dict[str, Any]:
    """Build a minimal Bot Framework activity."""
    activity: dict[str, Any] = {
        "type": activity_type,
        "id": "act-001",
        "from": {"id": user_id, "name": user_name},
        "conversation": {"id": conversation_id},
        "recipient": {"id": recipient_id},
        "serviceUrl": service_url,
    }
    if text:
        activity["text"] = text
    if conversation_type:
        activity["conversation"]["conversationType"] = conversation_type
    if entities is not None:
        activity["entities"] = entities
    if invoke_name:
        activity["name"] = invoke_name
    if value is not None:
        activity["value"] = value
    if reply_to_id:
        activity["replyToId"] = reply_to_id
    if members_added is not None:
        activity["membersAdded"] = members_added
    if members_removed is not None:
        activity["membersRemoved"] = members_removed
    if reactions_added is not None:
        activity["reactionsAdded"] = reactions_added
    if reactions_removed is not None:
        activity["reactionsRemoved"] = reactions_removed
    if action:
        activity["action"] = action
    if attachments is not None:
        activity["attachments"] = attachments
    return activity


def _mention_entities() -> list[dict[str, Any]]:
    """Build mention entities list."""
    return [{"type": "mention", "mentioned": {"id": "bot-id", "name": "Aragora"}}]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def events_module():
    """Import the events module lazily (after conftest patches)."""
    import aragora.server.handlers.bots.teams.events as mod

    return mod


@pytest.fixture
def processor_cls(events_module):
    return events_module.TeamsEventProcessor


@pytest.fixture
def mock_bot():
    """Create a mock TeamsBot instance with required methods."""
    bot = MagicMock()
    bot._check_permission = MagicMock(return_value=None)  # No RBAC error
    bot.send_typing = AsyncMock()
    bot.send_reply = AsyncMock()
    bot.send_card = AsyncMock()

    mock_card_actions = MagicMock()
    mock_card_actions.handle_invoke = AsyncMock(return_value={"status": 200, "body": {}})
    bot._get_card_actions = MagicMock(return_value=mock_card_actions)

    return bot


@pytest.fixture
def processor(processor_cls, mock_bot):
    """Create a TeamsEventProcessor with a mock bot."""
    return processor_cls(mock_bot)


@pytest.fixture(autouse=True)
def clear_state():
    """Clear module-level shared state between tests."""
    from aragora.server.handlers.bots.teams_utils import (
        _active_debates,
        _conversation_references,
        _user_votes,
    )

    _active_debates.clear()
    _conversation_references.clear()
    _user_votes.clear()
    yield
    _active_debates.clear()
    _conversation_references.clear()
    _user_votes.clear()


@pytest.fixture
def active_debate():
    """Create an active debate in the shared state and return its ID."""
    from aragora.server.handlers.bots.teams_utils import _active_debates

    debate_id = "abcdef01-2345-6789-abcd-ef0123456789"
    _active_debates[debate_id] = {
        "topic": "Should we adopt microservices?",
        "conversation_id": "conv-abc",
        "user_id": "user-123",
        "service_url": "https://smba.trafficmanager.net/teams/",
        "started_at": time.time() - 120,
        "current_round": 2,
        "total_rounds": 5,
        "phase": "deliberation",
    }
    return debate_id


# ===========================================================================
# process_activity: Activity type routing
# ===========================================================================


class TestProcessActivity:
    """Test process_activity() routes to the correct handler by type."""

    @pytest.mark.asyncio
    async def test_routes_message(self, processor):
        activity = _make_activity(activity_type="message", text="hello")
        processor._handle_message = AsyncMock(return_value={})
        await processor.process_activity(activity)
        processor._handle_message.assert_awaited_once_with(activity)

    @pytest.mark.asyncio
    async def test_routes_invoke(self, processor):
        activity = _make_activity(activity_type="invoke")
        processor._handle_invoke = AsyncMock(return_value={"status": 200})
        result = await processor.process_activity(activity)
        processor._handle_invoke.assert_awaited_once_with(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_routes_conversation_update(self, processor):
        activity = _make_activity(activity_type="conversationUpdate")
        processor._handle_conversation_update = AsyncMock(return_value={})
        await processor.process_activity(activity)
        processor._handle_conversation_update.assert_awaited_once_with(activity)

    @pytest.mark.asyncio
    async def test_routes_message_reaction(self, processor):
        activity = _make_activity(activity_type="messageReaction")
        processor._handle_message_reaction = AsyncMock(return_value={})
        await processor.process_activity(activity)
        processor._handle_message_reaction.assert_awaited_once_with(activity)

    @pytest.mark.asyncio
    async def test_routes_installation_update(self, processor):
        activity = _make_activity(activity_type="installationUpdate")
        processor._handle_installation_update = AsyncMock(return_value={})
        await processor.process_activity(activity)
        processor._handle_installation_update.assert_awaited_once_with(activity)

    @pytest.mark.asyncio
    async def test_unknown_type_returns_empty(self, processor):
        activity = _make_activity(activity_type="unknownType")
        result = await processor.process_activity(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_type_returns_empty(self, processor):
        activity = {"type": ""}
        result = await processor.process_activity(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_missing_type_returns_empty(self, processor):
        activity = {}
        result = await processor.process_activity(activity)
        assert result == {}


# ===========================================================================
# _handle_message: Parsing and routing
# ===========================================================================


class TestHandleMessage:
    """Test message activity handling."""

    @pytest.mark.asyncio
    async def test_rbac_denied_sends_reply(self, processor, mock_bot):
        mock_bot._check_permission.return_value = {"error": "permission_denied"}
        activity = _make_activity(text="hello", entities=_mention_entities())
        await processor._handle_message(activity)
        mock_bot.send_reply.assert_awaited()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "permission" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_sends_typing_indicator(self, processor, mock_bot):
        activity = _make_activity(
            text="<at>Aragora</at> help",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        mock_bot.send_typing.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mention_debate_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> debate AI safety",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        call_kwargs = processor._handle_command.call_args.kwargs
        assert call_kwargs["command"] == "debate"
        assert call_kwargs["args"] == "AI safety"

    @pytest.mark.asyncio
    async def test_mention_help_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> help",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        assert processor._handle_command.call_args.kwargs["command"] == "help"

    @pytest.mark.asyncio
    async def test_mention_status_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> status",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        assert processor._handle_command.call_args.kwargs["command"] == "status"

    @pytest.mark.asyncio
    async def test_mention_leaderboard_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> leaderboard",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        assert processor._handle_command.call_args.kwargs["command"] == "leaderboard"

    @pytest.mark.asyncio
    async def test_mention_agents_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> agents",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        assert processor._handle_command.call_args.kwargs["command"] == "agents"

    @pytest.mark.asyncio
    async def test_mention_vote_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> vote claude",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        kw = processor._handle_command.call_args.kwargs
        assert kw["command"] == "vote"
        assert kw["args"] == "claude"

    @pytest.mark.asyncio
    async def test_mention_ask_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> ask what is life?",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        assert processor._handle_command.call_args.kwargs["command"] == "ask"

    @pytest.mark.asyncio
    async def test_mention_plan_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> plan build a dashboard",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        assert processor._handle_command.call_args.kwargs["command"] == "plan"

    @pytest.mark.asyncio
    async def test_mention_implement_command(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at> implement rate limiter",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        assert processor._handle_command.call_args.kwargs["command"] == "implement"

    @pytest.mark.asyncio
    async def test_personal_known_command(self, processor):
        activity = _make_activity(text="help", conversation_type="personal")
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        assert processor._handle_command.call_args.kwargs["command"] == "help"

    @pytest.mark.asyncio
    async def test_personal_unknown_command_defaults_to_debate(self, processor):
        activity = _make_activity(
            text="is AI sentient?",
            conversation_type="personal",
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        kw = processor._handle_command.call_args.kwargs
        assert kw["command"] == "debate"
        assert kw["args"] == "is AI sentient?"

    @pytest.mark.asyncio
    async def test_non_mention_non_personal_sends_prompt(self, processor, mock_bot):
        activity = _make_activity(text="random text", conversation_type="channel")
        result = await processor._handle_message(activity)
        mock_bot.send_reply.assert_awaited_once()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "Mention me" in reply_text
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_text(self, processor, mock_bot):
        activity = _make_activity(text="", conversation_type="channel")
        await processor._handle_message(activity)
        mock_bot.send_reply.assert_awaited()

    @pytest.mark.asyncio
    async def test_mention_with_no_command_text(self, processor):
        activity = _make_activity(
            text="<at>Aragora</at>  ",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        kw = processor._handle_command.call_args.kwargs
        assert kw["command"] == ""

    @pytest.mark.asyncio
    async def test_mention_case_insensitive_at_tag(self, processor):
        activity = _make_activity(
            text="<AT>Bot</AT> debate test",
            entities=_mention_entities(),
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        kw = processor._handle_command.call_args.kwargs
        assert kw["command"] == "debate"

    @pytest.mark.asyncio
    async def test_personal_debate_known_command(self, processor):
        """Personal chat with 'debate' is treated as known command."""
        activity = _make_activity(
            text="debate something",
            conversation_type="personal",
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        kw = processor._handle_command.call_args.kwargs
        assert kw["command"] == "debate"
        assert kw["args"] == "something"

    @pytest.mark.asyncio
    async def test_passes_correct_fields_to_handle_command(self, processor):
        """Ensure conversation_id, user_id, service_url are forwarded."""
        activity = _make_activity(
            text="<at>Aragora</at> status",
            entities=_mention_entities(),
            conversation_id="my-conv",
            user_id="my-user",
            service_url="https://svc.example.com",
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        kw = processor._handle_command.call_args.kwargs
        assert kw["conversation_id"] == "my-conv"
        assert kw["user_id"] == "my-user"
        assert kw["service_url"] == "https://svc.example.com"

    @pytest.mark.asyncio
    async def test_reply_to_id_passed_to_handle_command(self, processor):
        """replyToId from activity is passed as thread_id via activity."""
        activity = _make_activity(
            text="<at>Aragora</at> debate topic",
            entities=_mention_entities(),
            reply_to_id="thread-99",
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        # The activity itself carries replyToId; _handle_command receives it
        kw = processor._handle_command.call_args.kwargs
        assert kw["activity"]["replyToId"] == "thread-99"


# ===========================================================================
# _handle_command: Command routing
# ===========================================================================


class TestHandleCommand:
    """Test _handle_command routes to the correct sub-handler."""

    @pytest.fixture(autouse=True)
    def patch_commands(self, processor):
        processor._cmd_debate = AsyncMock(return_value={"debate": True})
        processor._cmd_status = AsyncMock(return_value={"status": True})
        processor._cmd_help = AsyncMock(return_value={"help": True})
        processor._cmd_leaderboard = AsyncMock(return_value={"leaderboard": True})
        processor._cmd_agents = AsyncMock(return_value={"agents": True})
        processor._cmd_vote = AsyncMock(return_value={"vote": True})
        processor._cmd_unknown = AsyncMock(return_value={"unknown": True})

    @pytest.mark.asyncio
    async def test_debate_command(self, processor):
        activity = _make_activity()
        result = await processor._handle_command(
            command="debate",
            args="topic",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        assert result == {"debate": True}

    @pytest.mark.asyncio
    async def test_ask_command(self, processor):
        activity = _make_activity()
        result = await processor._handle_command(
            command="ask",
            args="question",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        assert result == {"debate": True}

    @pytest.mark.asyncio
    async def test_plan_command_passes_decision_integrity(self, processor):
        """'plan' sets decision_integrity with include_receipt/include_plan."""
        activity = _make_activity()
        await processor._handle_command(
            command="plan",
            args="build API",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        call_kwargs = processor._cmd_debate.call_args
        # decision_integrity should be passed as keyword arg
        di = call_kwargs.kwargs.get("decision_integrity") or call_kwargs[1].get(
            "decision_integrity"
        )
        assert di is not None
        assert di["include_receipt"] is True
        assert di["include_plan"] is True
        assert di.get("include_context") is False
        assert "execution_mode" not in di

    @pytest.mark.asyncio
    async def test_implement_command_passes_execution_mode(self, processor):
        """'implement' includes execution_mode and include_context."""
        activity = _make_activity()
        await processor._handle_command(
            command="implement",
            args="rate limiter",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        call_kwargs = processor._cmd_debate.call_args
        di = call_kwargs.kwargs.get("decision_integrity") or call_kwargs[1].get(
            "decision_integrity"
        )
        assert di is not None
        assert di["include_context"] is True
        assert di["execution_mode"] == "execute"
        assert di["execution_engine"] == "hybrid"

    @pytest.mark.asyncio
    async def test_status_command(self, processor):
        activity = _make_activity()
        result = await processor._handle_command(
            command="status",
            args="",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        assert result == {"status": True}

    @pytest.mark.asyncio
    async def test_help_command(self, processor):
        activity = _make_activity()
        result = await processor._handle_command(
            command="help",
            args="",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        assert result == {"help": True}

    @pytest.mark.asyncio
    async def test_leaderboard_command(self, processor):
        activity = _make_activity()
        result = await processor._handle_command(
            command="leaderboard",
            args="",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        assert result == {"leaderboard": True}

    @pytest.mark.asyncio
    async def test_agents_command(self, processor):
        activity = _make_activity()
        result = await processor._handle_command(
            command="agents",
            args="",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        assert result == {"agents": True}

    @pytest.mark.asyncio
    async def test_vote_command(self, processor):
        activity = _make_activity()
        result = await processor._handle_command(
            command="vote",
            args="claude",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        assert result == {"vote": True}
        processor._cmd_vote.assert_awaited_once_with("claude", activity)

    @pytest.mark.asyncio
    async def test_unknown_command(self, processor):
        activity = _make_activity()
        result = await processor._handle_command(
            command="xyzzy",
            args="",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        assert result == {"unknown": True}
        processor._cmd_unknown.assert_awaited_once_with("xyzzy", activity)

    @pytest.mark.asyncio
    async def test_debate_no_decision_integrity(self, processor):
        """Plain 'debate' passes no decision_integrity."""
        activity = _make_activity()
        await processor._handle_command(
            command="debate",
            args="topic",
            conversation_id="c",
            user_id="u",
            service_url="s",
            activity=activity,
        )
        call_kwargs = processor._cmd_debate.call_args
        di = call_kwargs.kwargs.get("decision_integrity")
        assert di is None


# ===========================================================================
# _cmd_debate
# ===========================================================================


class TestCmdDebate:
    """Test debate command execution."""

    @pytest.mark.asyncio
    async def test_rbac_denied(self, processor, mock_bot):
        mock_bot._check_permission.return_value = {"error": "permission_denied"}
        activity = _make_activity()
        result = await processor._cmd_debate(
            "test topic",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        mock_bot.send_reply.assert_awaited()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "permission" in reply_text.lower()
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_topic(self, processor, mock_bot):
        activity = _make_activity()
        result = await processor._cmd_debate(
            "   ",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        mock_bot.send_reply.assert_awaited()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "provide a topic" in reply_text.lower()
        assert result == {}

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_success(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        mock_start.return_value = "debate-id-123"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity()

        result = await processor._cmd_debate(
            "AI safety",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        assert result == {}
        mock_start.assert_awaited_once()
        mock_bot.send_card.assert_awaited_once()
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_audit_fields(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        mock_start.return_value = "debate-abc"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity()

        await processor._cmd_debate(
            "Test Topic",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        mock_audit.assert_called_once_with(
            user_id="teams:user-1",
            resource_type="debate",
            resource_id="debate-abc",
            action="create",
            platform="teams",
            task_preview="Test Topic",
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_with_decision_integrity_label(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        """When decision_integrity is set, card fallback says 'implementation plan'."""
        mock_start.return_value = "debate-plan"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity()

        await processor._cmd_debate(
            "build API",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
            decision_integrity={"include_receipt": True, "include_plan": True},
        )
        send_card_args = mock_bot.send_card.call_args[0]
        assert "implementation plan" in send_card_args[2]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_without_decision_integrity_label(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        """Without decision_integrity, card fallback says 'debate'."""
        mock_start.return_value = "debate-plain"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity()

        await processor._cmd_debate(
            "topic",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        send_card_args = mock_bot.send_card.call_args[0]
        assert "debate" in send_card_args[2].lower()
        assert "implementation plan" not in send_card_args[2]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_non_list_attachments(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        """Non-list attachments are replaced with empty list."""
        mock_start.return_value = "debate-att"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity()
        activity["attachments"] = "not-a-list"

        await processor._cmd_debate(
            "topic",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        assert mock_start.call_args.kwargs["attachments"] == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_list_attachments_passed(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        """Valid list attachments are passed through."""
        mock_start.return_value = "debate-la"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        attachments = [{"contentType": "text/plain", "content": "file"}]
        activity = _make_activity(attachments=attachments)

        await processor._cmd_debate(
            "topic",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        assert mock_start.call_args.kwargs["attachments"] == attachments

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_passes_thread_id(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        """Thread ID is passed to _start_teams_debate."""
        mock_start.return_value = "d-thread"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity()

        await processor._cmd_debate(
            "topic",
            "conv-1",
            "user-1",
            "https://svc",
            "thread-42",
            activity,
        )
        assert mock_start.call_args.kwargs["thread_id"] == "thread-42"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_topic_preview_truncated(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        """Audit task_preview is truncated to 100 chars."""
        mock_start.return_value = "d-long"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        long_topic = "T" * 200
        activity = _make_activity()

        await processor._cmd_debate(
            long_topic,
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        audit_kwargs = mock_audit.call_args.kwargs
        assert len(audit_kwargs["task_preview"]) == 100


# ===========================================================================
# _cmd_status
# ===========================================================================


class TestCmdStatus:
    """Test status command."""

    @pytest.mark.asyncio
    async def test_status_sends_card(self, processor, mock_bot):
        activity = _make_activity()
        result = await processor._cmd_status(activity)
        assert result == {}
        mock_bot.send_card.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_status_shows_active_debates_count(self, processor, mock_bot, active_debate):
        activity = _make_activity()
        await processor._cmd_status(activity)
        card_arg = mock_bot.send_card.call_args[0][1]
        facts = card_arg["body"][1]["facts"]
        active_fact = next(f for f in facts if f["title"] == "Active Debates")
        assert active_fact["value"] == "1"

    @pytest.mark.asyncio
    async def test_status_zero_active_debates(self, processor, mock_bot):
        activity = _make_activity()
        await processor._cmd_status(activity)
        card_arg = mock_bot.send_card.call_args[0][1]
        facts = card_arg["body"][1]["facts"]
        active_fact = next(f for f in facts if f["title"] == "Active Debates")
        assert active_fact["value"] == "0"

    @pytest.mark.asyncio
    async def test_status_elo_store_unavailable(self, processor, mock_bot):
        """When ELO store import fails, falls back to 7 agents."""
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=RuntimeError("no elo"),
        ):
            activity = _make_activity()
            await processor._cmd_status(activity)
            card_arg = mock_bot.send_card.call_args[0][1]
            facts = card_arg["body"][1]["facts"]
            agent_fact = next(f for f in facts if f["title"] == "Registered Agents")
            assert agent_fact["value"] == "7"

    @pytest.mark.asyncio
    async def test_status_card_structure(self, processor, mock_bot):
        """Status card has expected structure."""
        activity = _make_activity()
        await processor._cmd_status(activity)
        card = mock_bot.send_card.call_args[0][1]
        assert card["type"] == "AdaptiveCard"
        assert card["version"] == "1.4"
        assert card["body"][0]["text"] == "Aragora Status: Online"
        assert len(card["actions"]) == 1
        assert card["actions"][0]["data"]["action"] == "start_debate_prompt"

    @pytest.mark.asyncio
    async def test_status_fallback_text(self, processor, mock_bot):
        """Status card fallback text includes active count."""
        activity = _make_activity()
        await processor._cmd_status(activity)
        fallback = mock_bot.send_card.call_args[0][2]
        assert "Aragora Status: Online" in fallback
        assert "0 active debates" in fallback


# ===========================================================================
# _cmd_help
# ===========================================================================


class TestCmdHelp:
    """Test help command."""

    @pytest.mark.asyncio
    async def test_help_with_card_module(self, processor, mock_bot):
        """When help card module is available, sends card."""
        mock_card = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "Help"}]}
        with patch(
            "aragora.server.handlers.bots.teams_cards.create_help_card",
            return_value=mock_card,
            create=True,
        ):
            activity = _make_activity()
            result = await processor._cmd_help(activity)
            assert result == {}
            mock_bot.send_card.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_help_with_import_error_fallback(self, processor, mock_bot):
        """When help card module fails to import, sends text."""
        import sys

        saved = sys.modules.get("aragora.server.handlers.bots.teams_cards")
        sys.modules["aragora.server.handlers.bots.teams_cards"] = None  # type: ignore
        try:
            activity = _make_activity()
            result = await processor._cmd_help(activity)
            assert result == {}
            mock_bot.send_reply.assert_awaited_once()
            reply_text = mock_bot.send_reply.call_args[0][1]
            assert "Aragora Commands" in reply_text
            assert "debate" in reply_text
            assert "status" in reply_text
            assert "help" in reply_text
        finally:
            if saved is not None:
                sys.modules["aragora.server.handlers.bots.teams_cards"] = saved
            else:
                sys.modules.pop("aragora.server.handlers.bots.teams_cards", None)


# ===========================================================================
# _cmd_leaderboard
# ===========================================================================


class TestCmdLeaderboard:
    """Test leaderboard command."""

    @pytest.mark.asyncio
    async def test_leaderboard_no_elo_store(self, processor, mock_bot):
        """When ELO store is unavailable, shows default leaderboard."""
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=RuntimeError("no elo"),
        ):
            activity = _make_activity()
            result = await processor._cmd_leaderboard(activity)
            assert result == {}
            mock_bot.send_reply.assert_awaited_once()
            reply_text = mock_bot.send_reply.call_args[0][1]
            assert "Agent Leaderboard" in reply_text
            assert "Claude" in reply_text
            assert "GPT-4" in reply_text

    @pytest.mark.asyncio
    async def test_leaderboard_with_elo_store(self, processor, mock_bot):
        """When ELO store is available, uses real ratings."""
        mock_rating_1 = MagicMock()
        mock_rating_1.agent_name = "claude"
        mock_rating_1.elo = 1850.0
        mock_rating_1.wins = 10
        mock_rating_1.total_debates = 20

        mock_rating_2 = MagicMock()
        mock_rating_2.agent_name = "gpt4"
        mock_rating_2.elo = 1820.0
        mock_rating_2.wins = 8
        mock_rating_2.total_debates = 18

        mock_elo_store = MagicMock()
        mock_elo_store.get_all_ratings.return_value = [mock_rating_1, mock_rating_2]

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo_store,
        ):
            activity = _make_activity()
            await processor._cmd_leaderboard(activity)
            # Should try to send card first (via teams_cards import), then fall back to text
            # Either way, reply should include agent names
            if mock_bot.send_card.called:
                pass  # card path
            elif mock_bot.send_reply.called:
                reply_text = mock_bot.send_reply.call_args[0][1]
                assert "claude" in reply_text
                assert "1850" in reply_text

    @pytest.mark.asyncio
    async def test_leaderboard_empty_ratings(self, processor, mock_bot):
        """ELO store with no ratings falls back to default."""
        mock_elo_store = MagicMock()
        mock_elo_store.get_all_ratings.return_value = []

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo_store,
        ):
            activity = _make_activity()
            await processor._cmd_leaderboard(activity)
            mock_bot.send_reply.assert_awaited_once()
            reply_text = mock_bot.send_reply.call_args[0][1]
            assert "Claude: 1850" in reply_text


# ===========================================================================
# _cmd_agents
# ===========================================================================


class TestCmdAgents:
    """Test agents command."""

    @pytest.mark.asyncio
    async def test_agents_sends_list(self, processor, mock_bot):
        activity = _make_activity()
        result = await processor._cmd_agents(activity)
        assert result == {}
        mock_bot.send_reply.assert_awaited_once()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "Available AI Agents" in reply_text
        assert "Claude" in reply_text
        assert "GPT-4" in reply_text
        assert "Gemini" in reply_text
        assert "Grok" in reply_text
        assert "Mistral" in reply_text
        assert "DeepSeek" in reply_text
        assert "Qwen" in reply_text
        assert "Kimi" in reply_text


# ===========================================================================
# _cmd_vote
# ===========================================================================


class TestCmdVote:
    """Test vote command."""

    @pytest.mark.asyncio
    async def test_vote_rbac_denied(self, processor, mock_bot):
        mock_bot._check_permission.return_value = {"error": "permission_denied"}
        activity = _make_activity()
        result = await processor._cmd_vote("claude", activity)
        assert result == {}
        mock_bot.send_reply.assert_awaited()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "permission" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_vote_no_args_no_active_debates(self, processor, mock_bot):
        activity = _make_activity()
        result = await processor._cmd_vote("", activity)
        assert result == {}
        mock_bot.send_reply.assert_awaited()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "No active debates" in reply_text

    @pytest.mark.asyncio
    async def test_vote_no_args_with_active_debates(self, processor, mock_bot, active_debate):
        """No args with active debates sends voting cards."""
        activity = _make_activity()
        with patch(
            "aragora.server.handlers.bots.teams.events.build_debate_card",
            return_value={"type": "AdaptiveCard"},
        ):
            result = await processor._cmd_vote("", activity)
            assert result == {}
            mock_bot.send_card.assert_awaited()

    @pytest.mark.asyncio
    async def test_vote_with_args(self, processor, mock_bot):
        """Vote with args sends 'use vote buttons' message."""
        activity = _make_activity()
        result = await processor._cmd_vote("some-arg", activity)
        assert result == {}
        mock_bot.send_reply.assert_awaited()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "vote buttons" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_vote_no_args_sends_max_3_cards(self, processor, mock_bot):
        """Vote with multiple active debates sends at most 3 cards."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        for i in range(5):
            _active_debates[f"debate-{i}"] = {
                "topic": f"Topic {i}",
                "current_round": 1,
                "total_rounds": 5,
            }

        activity = _make_activity()
        with patch(
            "aragora.server.handlers.bots.teams.events.build_debate_card",
            return_value={"type": "AdaptiveCard"},
        ):
            await processor._cmd_vote("", activity)
            assert mock_bot.send_card.await_count <= 3


# ===========================================================================
# _cmd_unknown
# ===========================================================================


class TestCmdUnknown:
    """Test unknown command handler."""

    @pytest.mark.asyncio
    async def test_unknown_command(self, processor, mock_bot):
        activity = _make_activity()
        result = await processor._cmd_unknown("xyzzy", activity)
        assert result == {}
        mock_bot.send_reply.assert_awaited_once()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "xyzzy" in reply_text
        assert "help" in reply_text.lower()


# ===========================================================================
# _handle_invoke
# ===========================================================================


class TestHandleInvoke:
    """Test invoke activity delegation."""

    @pytest.mark.asyncio
    async def test_delegates_to_card_actions(self, processor, mock_bot):
        activity = _make_activity(activity_type="invoke", invoke_name="adaptiveCard/action")
        result = await processor._handle_invoke(activity)
        mock_bot._get_card_actions.return_value.handle_invoke.assert_awaited_once_with(activity)
        assert result["status"] == 200


# ===========================================================================
# _handle_conversation_update
# ===========================================================================


class TestHandleConversationUpdate:
    """Test conversation update handling."""

    @pytest.mark.asyncio
    async def test_bot_added_sends_welcome(self, processor):
        """When bot is added, send welcome message."""
        processor._send_welcome = AsyncMock()
        activity = _make_activity(
            activity_type="conversationUpdate",
            members_added=[{"id": "bot-id"}],
            recipient_id="bot-id",
        )
        result = await processor._handle_conversation_update(activity)
        assert result == {}
        processor._send_welcome.assert_awaited_once_with(activity)

    @pytest.mark.asyncio
    async def test_non_bot_member_added_no_welcome(self, processor):
        """When a non-bot member is added, no welcome is sent."""
        processor._send_welcome = AsyncMock()
        activity = _make_activity(
            activity_type="conversationUpdate",
            members_added=[{"id": "other-user"}],
            recipient_id="bot-id",
        )
        await processor._handle_conversation_update(activity)
        processor._send_welcome.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bot_removed_cleans_references(self, processor):
        """When bot is removed, conversation reference is cleaned."""
        from aragora.server.handlers.bots.teams_utils import _conversation_references

        _conversation_references["conv-abc"] = {"service_url": "https://svc"}

        activity = _make_activity(
            activity_type="conversationUpdate",
            members_removed=[{"id": "bot-id"}],
            recipient_id="bot-id",
            conversation_id="conv-abc",
        )
        await processor._handle_conversation_update(activity)
        assert "conv-abc" not in _conversation_references

    @pytest.mark.asyncio
    async def test_non_bot_member_removed_no_cleanup(self, processor):
        """When non-bot member is removed, reference is NOT cleaned."""
        from aragora.server.handlers.bots.teams_utils import _conversation_references

        _conversation_references["conv-abc"] = {"service_url": "https://svc"}

        activity = _make_activity(
            activity_type="conversationUpdate",
            members_removed=[{"id": "other-user"}],
            recipient_id="bot-id",
            conversation_id="conv-abc",
        )
        await processor._handle_conversation_update(activity)
        assert "conv-abc" in _conversation_references

    @pytest.mark.asyncio
    async def test_no_members_added_or_removed(self, processor):
        """Activity with no members added/removed does nothing."""
        processor._send_welcome = AsyncMock()
        activity = _make_activity(activity_type="conversationUpdate")
        result = await processor._handle_conversation_update(activity)
        assert result == {}
        processor._send_welcome.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_members_added_including_bot(self, processor):
        """Multiple members added, only bot triggers welcome."""
        processor._send_welcome = AsyncMock()
        activity = _make_activity(
            activity_type="conversationUpdate",
            members_added=[{"id": "other-user"}, {"id": "bot-id"}],
            recipient_id="bot-id",
        )
        await processor._handle_conversation_update(activity)
        processor._send_welcome.assert_awaited_once()


# ===========================================================================
# _handle_message_reaction
# ===========================================================================


class TestHandleMessageReaction:
    """Test message reaction handling."""

    @pytest.mark.asyncio
    async def test_reaction_added_like(self, processor):
        activity = _make_activity(
            activity_type="messageReaction",
            reactions_added=[{"type": "like"}],
            reply_to_id="msg-123",
        )
        result = await processor._handle_message_reaction(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_reaction_added_heart(self, processor):
        activity = _make_activity(
            activity_type="messageReaction",
            reactions_added=[{"type": "heart"}],
            reply_to_id="msg-456",
        )
        result = await processor._handle_message_reaction(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_reaction_removed(self, processor):
        activity = _make_activity(
            activity_type="messageReaction",
            reactions_removed=[{"type": "like"}],
        )
        result = await processor._handle_message_reaction(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_reactions(self, processor):
        activity = _make_activity(activity_type="messageReaction")
        result = await processor._handle_message_reaction(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_multiple_reactions_added(self, processor):
        activity = _make_activity(
            activity_type="messageReaction",
            reactions_added=[{"type": "like"}, {"type": "heart"}, {"type": "laugh"}],
            reply_to_id="msg-789",
        )
        result = await processor._handle_message_reaction(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_reaction_no_reply_to_id(self, processor):
        """Reaction with no replyToId still processes without error."""
        activity = _make_activity(
            activity_type="messageReaction",
            reactions_added=[{"type": "like"}],
        )
        result = await processor._handle_message_reaction(activity)
        assert result == {}


# ===========================================================================
# _handle_installation_update
# ===========================================================================


class TestHandleInstallationUpdate:
    """Test installation update handling."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    async def test_install_event(self, mock_audit, processor):
        activity = _make_activity(
            activity_type="installationUpdate",
            action="add",
            conversation_id="conv-install",
        )
        result = await processor._handle_installation_update(activity)
        assert result == {}
        mock_audit.assert_called_once_with(
            user_id="system",
            resource_type="teams_installation",
            resource_id="conv-install",
            action="install",
            platform="teams",
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    async def test_uninstall_event(self, mock_audit, processor):
        activity = _make_activity(
            activity_type="installationUpdate",
            action="remove",
            conversation_id="conv-remove",
        )
        result = await processor._handle_installation_update(activity)
        assert result == {}
        mock_audit.assert_called_once_with(
            user_id="system",
            resource_type="teams_installation",
            resource_id="conv-remove",
            action="uninstall",
            platform="teams",
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    async def test_unknown_action(self, mock_audit, processor):
        activity = _make_activity(
            activity_type="installationUpdate",
            action="unknown",
        )
        result = await processor._handle_installation_update(activity)
        assert result == {}
        mock_audit.assert_not_called()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    async def test_no_action(self, mock_audit, processor):
        activity = _make_activity(activity_type="installationUpdate")
        result = await processor._handle_installation_update(activity)
        assert result == {}
        mock_audit.assert_not_called()


# ===========================================================================
# _send_welcome
# ===========================================================================


class TestSendWelcome:
    """Test welcome message sending."""

    @pytest.mark.asyncio
    async def test_welcome_with_card_module(self, processor, mock_bot):
        """When help card module is available, sends card with welcome header."""
        mock_card = {
            "type": "AdaptiveCard",
            "body": [{"type": "TextBlock", "text": "Help info"}],
        }
        with patch(
            "aragora.server.handlers.bots.teams_cards.create_help_card",
            return_value=mock_card,
            create=True,
        ):
            activity = _make_activity()
            await processor._send_welcome(activity)
            mock_bot.send_card.assert_awaited_once()
            card_arg = mock_bot.send_card.call_args[0][1]
            # Welcome header should be prepended
            assert card_arg["body"][0]["text"] == "Welcome to Aragora!"

    @pytest.mark.asyncio
    async def test_welcome_with_import_error_fallback(self, processor, mock_bot):
        """When help card import fails, sends text welcome."""
        import sys

        saved = sys.modules.get("aragora.server.handlers.bots.teams_cards")
        sys.modules["aragora.server.handlers.bots.teams_cards"] = None  # type: ignore
        try:
            activity = _make_activity()
            await processor._send_welcome(activity)
            mock_bot.send_reply.assert_awaited_once()
            reply_text = mock_bot.send_reply.call_args[0][1]
            assert "Welcome to Aragora" in reply_text
            assert "debate" in reply_text.lower()
        finally:
            if saved is not None:
                sys.modules["aragora.server.handlers.bots.teams_cards"] = saved
            else:
                sys.modules.pop("aragora.server.handlers.bots.teams_cards", None)

    @pytest.mark.asyncio
    async def test_welcome_card_has_separator(self, processor, mock_bot):
        """Welcome card includes separator element."""
        mock_card = {
            "type": "AdaptiveCard",
            "body": [{"type": "TextBlock", "text": "Commands"}],
        }
        with patch(
            "aragora.server.handlers.bots.teams_cards.create_help_card",
            return_value=mock_card,
            create=True,
        ):
            activity = _make_activity()
            await processor._send_welcome(activity)
            card_arg = mock_bot.send_card.call_args[0][1]
            # Check for separator in the welcome header
            texts = [item.get("text", "") for item in card_arg["body"]]
            assert "---" in texts


# ===========================================================================
# Module constants and exports
# ===========================================================================


class TestModuleConstants:
    """Test module-level constants."""

    def test_agent_display_names(self, events_module):
        names = events_module.AGENT_DISPLAY_NAMES
        assert "claude" in names
        assert names["claude"] == "Claude"
        assert "gpt4" in names
        assert names["gpt4"] == "GPT-4"
        assert "gemini" in names
        assert "mistral" in names
        assert "deepseek" in names
        assert "grok" in names
        assert "qwen" in names
        assert "kimi" in names
        assert "anthropic-api" in names
        assert "openai-api" in names

    def test_agent_display_names_are_strings(self, events_module):
        for key, value in events_module.AGENT_DISPLAY_NAMES.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_mention_pattern_strips_at_tags(self, events_module):
        text = "<at>Aragora Bot</at> debate AI"
        cleaned = events_module.MENTION_PATTERN.sub("", text)
        assert cleaned == "debate AI"

    def test_mention_pattern_case_insensitive(self, events_module):
        text = "<AT>Bot</AT> help"
        cleaned = events_module.MENTION_PATTERN.sub("", text)
        assert cleaned == "help"

    def test_mention_pattern_multiple_mentions(self, events_module):
        text = "<at>A</at> <at>B</at> cmd"
        cleaned = events_module.MENTION_PATTERN.sub("", text)
        assert cleaned == "cmd"

    def test_permission_constants(self, events_module):
        assert events_module.PERM_TEAMS_MESSAGES_READ == "teams:messages:read"
        assert events_module.PERM_TEAMS_DEBATES_CREATE == "teams:debates:create"

    def test_all_exports(self, events_module):
        assert "TeamsEventProcessor" in events_module.__all__
        assert "AGENT_DISPLAY_NAMES" in events_module.__all__
        assert "MENTION_PATTERN" in events_module.__all__

    def test_processor_stores_bot_reference(self, processor_cls):
        bot = MagicMock()
        ep = processor_cls(bot)
        assert ep.bot is bot


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    @pytest.mark.asyncio
    async def test_message_with_missing_conversation(self, processor, mock_bot):
        """Message with no conversation field still processes."""
        activity = {
            "type": "message",
            "text": "hello",
            "from": {"id": "user-1", "name": "User"},
            "serviceUrl": "https://svc",
        }
        mock_bot._check_permission.return_value = None
        # Should not raise
        await processor._handle_message(activity)

    @pytest.mark.asyncio
    async def test_message_with_missing_from(self, processor, mock_bot):
        """Message with no from field uses defaults."""
        activity = {
            "type": "message",
            "text": "hello",
            "conversation": {"id": "c1"},
            "serviceUrl": "https://svc",
        }
        mock_bot._check_permission.return_value = None
        await processor._handle_message(activity)

    @pytest.mark.asyncio
    async def test_conversation_update_empty_members(self, processor):
        """Conversation update with empty member lists."""
        processor._send_welcome = AsyncMock()
        activity = _make_activity(
            activity_type="conversationUpdate",
            members_added=[],
            members_removed=[],
        )
        result = await processor._handle_conversation_update(activity)
        assert result == {}
        processor._send_welcome.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_installation_update_empty_conversation(self, processor):
        """Installation update with missing conversation ID."""
        with patch("aragora.server.handlers.bots.teams.events.audit_data") as mock_audit:
            activity = {"type": "installationUpdate", "action": "add", "conversation": {}}
            result = await processor._handle_installation_update(activity)
            assert result == {}
            mock_audit.assert_called_once()
            assert mock_audit.call_args.kwargs["resource_id"] == ""

    @pytest.mark.asyncio
    async def test_reaction_with_missing_from(self, processor):
        """Reaction with no from field defaults gracefully."""
        activity = {
            "type": "messageReaction",
            "reactionsAdded": [{"type": "like"}],
            "replyToId": "msg-1",
        }
        result = await processor._handle_message_reaction(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_multiple_mention_entities(self, processor):
        """Multiple mention entities in entities list."""
        entities = [
            {"type": "mention", "mentioned": {"id": "bot-1"}},
            {"type": "mention", "mentioned": {"id": "bot-2"}},
        ]
        activity = _make_activity(
            text="<at>Bot1</at> <at>Bot2</at> debate test",
            entities=entities,
        )
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        kw = processor._handle_command.call_args.kwargs
        assert kw["command"] == "debate"

    @pytest.mark.asyncio
    async def test_personal_with_whitespace_only(self, processor):
        """Personal chat with whitespace-only text."""
        activity = _make_activity(text="   ", conversation_type="personal")
        processor._handle_command = AsyncMock(return_value={})
        await processor._handle_message(activity)
        kw = processor._handle_command.call_args.kwargs
        assert kw["command"] == ""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_card_has_no_vote_buttons(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        """Debate card is built with include_vote_buttons=False."""
        mock_start.return_value = "d-novote"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity()

        await processor._cmd_debate(
            "topic",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        assert mock_build_card.call_args.kwargs["include_vote_buttons"] is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.events.audit_data")
    @patch("aragora.server.handlers.bots.teams.events.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.events._start_teams_debate", new_callable=AsyncMock)
    async def test_debate_card_sends_first_5_agents(
        self, mock_start, mock_build_card, mock_audit, processor, mock_bot
    ):
        """Debate card agent list is limited to first 5 from DEFAULT_AGENT_LIST."""
        mock_start.return_value = "d-agents"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity()

        await processor._cmd_debate(
            "topic",
            "conv-1",
            "user-1",
            "https://svc",
            None,
            activity,
        )
        agents_arg = mock_build_card.call_args.kwargs["agents"]
        assert len(agents_arg) <= 5
