"""
Tests for Teams Bot Adaptive Card action handling.

Covers all routes and behavior of the TeamsCardActions class:
- handle_invoke() routing for all invoke names
- Card action routing (_handle_card_action) for all action types
- Vote actions (new vote, vote change, missing data, RBAC)
- Summary actions (active debate, missing debate)
- View details (with/without progress card import)
- View report
- View rankings
- Watch debate (subscribe, already watching, missing debate)
- Share result (active/missing debate)
- Start debate prompt (task module card)
- Help action (no RBAC check)
- Compose extension submit (startDebate, unknown command, RBAC, missing topic)
- Compose extension query (search, empty query, RBAC, result limit)
- Task module fetch (startDebate, generic, RBAC)
- Task module submit (start debate from task module, empty topic, RBAC)
- Link unfurling (valid URL, no match, RBAC)
- Unknown/unhandled card actions
- Default invoke fallthrough
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
    user_id: str = "user-123",
    invoke_name: str = "",
    value: dict[str, Any] | None = None,
    conversation_id: str = "conv-abc",
    service_url: str = "https://smba.trafficmanager.net/teams/",
    aad_object_id: str = "",
    tenant_id: str = "",
    attachments: list | None = None,
) -> dict[str, Any]:
    """Build a minimal Bot Framework invoke activity."""
    activity: dict[str, Any] = {
        "type": "invoke",
        "from": {"id": user_id},
        "conversation": {"id": conversation_id},
        "serviceUrl": service_url,
    }
    if invoke_name:
        activity["name"] = invoke_name
    if value is not None:
        activity["value"] = value
    if aad_object_id:
        activity["from"]["aadObjectId"] = aad_object_id
    if tenant_id:
        activity["conversation"]["tenantId"] = tenant_id
    if attachments is not None:
        activity["attachments"] = attachments
    return activity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_bot():
    """Create a mock TeamsBot instance with required methods."""
    bot = MagicMock()
    bot._check_permission = MagicMock(return_value=None)  # No RBAC error
    bot.send_reply = AsyncMock()
    bot.send_card = AsyncMock()

    mock_event_processor = MagicMock()
    mock_event_processor._cmd_help = AsyncMock()
    mock_event_processor._cmd_leaderboard = AsyncMock()
    bot._get_event_processor = MagicMock(return_value=mock_event_processor)

    return bot


@pytest.fixture
def card_actions(mock_bot):
    """Create a TeamsCardActions instance with a mock bot."""
    from aragora.server.handlers.bots.teams.cards import TeamsCardActions

    return TeamsCardActions(mock_bot)


@pytest.fixture(autouse=True)
def clear_state():
    """Clear module-level shared state between tests."""
    from aragora.server.handlers.bots.teams_utils import _active_debates, _user_votes

    _active_debates.clear()
    _user_votes.clear()
    yield
    _active_debates.clear()
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
        "started_at": time.time() - 120,  # 2 minutes ago
        "current_round": 2,
        "total_rounds": 5,
        "phase": "deliberation",
        "consensus_reached": True,
        "confidence": 0.85,
        "winner": "claude",
        "final_answer": "Microservices offer scalability benefits.",
    }
    return debate_id


# ===========================================================================
# handle_invoke routing tests
# ===========================================================================


class TestHandleInvokeRouting:
    """Test that handle_invoke routes to the correct handler."""

    @pytest.mark.asyncio
    async def test_adaptive_card_action(self, card_actions):
        """adaptiveCard/action routes to _handle_card_action."""
        activity = _make_activity(invoke_name="adaptiveCard/action", value={"action": ""})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_empty_invoke_name_routes_to_card_action(self, card_actions):
        """Empty invoke name routes to _handle_card_action."""
        activity = _make_activity(invoke_name="", value={"action": ""})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_no_invoke_name_routes_to_card_action(self, card_actions):
        """Missing invoke name routes to _handle_card_action."""
        activity = _make_activity(value={"action": ""})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_compose_extension_submit(self, card_actions):
        """composeExtension/submitAction routes correctly."""
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "unknown"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_compose_extension_query(self, card_actions):
        """composeExtension/query routes correctly."""
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": []},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_compose_extension_fetch_task(self, card_actions):
        """composeExtension/fetchTask routes to task module fetch."""
        activity = _make_activity(
            invoke_name="composeExtension/fetchTask",
            value={},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_task_fetch(self, card_actions):
        """task/fetch routes to task module fetch."""
        activity = _make_activity(
            invoke_name="task/fetch",
            value={},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_task_submit(self, card_actions):
        """task/submit routes to task module submit."""
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_query_link(self, card_actions):
        """composeExtension/queryLink routes to link unfurling."""
        activity = _make_activity(
            invoke_name="composeExtension/queryLink",
            value={"url": "https://example.com"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_unknown_invoke_name(self, card_actions):
        """Unknown invoke name returns default 200 response."""
        activity = _make_activity(invoke_name="unknown/action", value={})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["type"] == "message"
        assert result["body"]["value"] == "Action processed"

    @pytest.mark.asyncio
    async def test_invoke_extracts_user_id(self, card_actions):
        """handle_invoke extracts user_id from activity."""
        activity = _make_activity(
            user_id="specific-user",
            invoke_name="adaptiveCard/action",
            value={"action": ""},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200


# ===========================================================================
# Card action routing tests
# ===========================================================================


class TestCardActionRouting:
    """Test _handle_card_action routes to the correct sub-handler."""

    @pytest.mark.asyncio
    async def test_unknown_action_acknowledged(self, card_actions):
        """Unknown action returns acknowledged response."""
        activity = _make_activity(value={"action": "nonexistent_action"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["value"] == "Action acknowledged"

    @pytest.mark.asyncio
    async def test_empty_action_acknowledged(self, card_actions):
        """Empty action returns acknowledged response."""
        activity = _make_activity(value={"action": ""})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["value"] == "Action acknowledged"

    @pytest.mark.asyncio
    async def test_card_action_rbac_permission_denied(self, card_actions, mock_bot):
        """Non-help actions check RBAC and return 403 on denial."""
        mock_bot._check_permission.return_value = {"error": "permission_denied"}
        activity = _make_activity(value={"action": "vote", "debate_id": "d1", "agent": "claude"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 403
        assert result["body"]["statusCode"] == 403
        body_items = result["body"]["value"]["body"]
        assert any("Permission Denied" in item.get("text", "") for item in body_items)

    @pytest.mark.asyncio
    async def test_help_action_skips_rbac(self, card_actions, mock_bot):
        """Help action skips RBAC permission check entirely."""
        mock_bot._check_permission.return_value = {"error": "permission_denied"}
        activity = _make_activity(value={"action": "help"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["value"] == "Help sent"

    @pytest.mark.asyncio
    async def test_help_action_calls_event_processor(self, card_actions, mock_bot):
        """Help action delegates to event processor _cmd_help."""
        activity = _make_activity(value={"action": "help"})
        await card_actions.handle_invoke(activity)
        mock_bot._get_event_processor.return_value._cmd_help.assert_called_once_with(activity)


# ===========================================================================
# Vote action tests
# ===========================================================================


class TestVoteAction:
    """Test _handle_vote."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_success(self, mock_audit, card_actions, active_debate):
        """Successful vote records and returns confirmation."""
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "claude"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        body_items = result["body"]["value"]["body"]
        assert any("recorded" in item.get("text", "") for item in body_items)

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_records_in_user_votes(self, mock_audit, card_actions, active_debate):
        """Vote is stored in _user_votes dict."""
        from aragora.server.handlers.bots.teams_utils import _user_votes

        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "gpt4"},
        )
        await card_actions.handle_invoke(activity)
        assert _user_votes[active_debate]["user-123"] == "gpt4"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_change_message(self, mock_audit, card_actions, active_debate):
        """Changing vote shows 'changed from X to Y' message."""
        from aragora.server.handlers.bots.teams_utils import _user_votes

        _user_votes[active_debate] = {"user-123": "claude"}
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "gpt4"},
        )
        result = await card_actions.handle_invoke(activity)
        body_items = result["body"]["value"]["body"]
        text_items = [item.get("text", "") for item in body_items]
        assert any("changed from claude to gpt4" in t for t in text_items)

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_same_agent_no_change_message(self, mock_audit, card_actions, active_debate):
        """Voting for same agent again shows recorded (not changed) message."""
        from aragora.server.handlers.bots.teams_utils import _user_votes

        _user_votes[active_debate] = {"user-123": "claude"}
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "claude"},
        )
        result = await card_actions.handle_invoke(activity)
        body_items = result["body"]["value"]["body"]
        text_items = [item.get("text", "") for item in body_items]
        assert any("recorded" in t for t in text_items)

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_shows_total_count(self, mock_audit, card_actions, active_debate):
        """Vote response includes total vote count."""
        from aragora.server.handlers.bots.teams_utils import _user_votes

        _user_votes[active_debate] = {"other-user": "gemini"}
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "claude"},
        )
        result = await card_actions.handle_invoke(activity)
        body_items = result["body"]["value"]["body"]
        text_items = [item.get("text", "") for item in body_items]
        assert any("Total votes cast: 2" in t for t in text_items)

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_audit_logged(self, mock_audit, card_actions, active_debate):
        """Vote action logs an audit event."""
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "claude"},
        )
        await card_actions.handle_invoke(activity)
        mock_audit.assert_called_once_with(
            user_id="teams:user-123",
            resource_type="debate_vote",
            resource_id=active_debate,
            action="create",
            vote_option="claude",
            platform="teams",
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_missing_debate_id(self, mock_audit, card_actions):
        """Vote with empty debate_id returns 400."""
        activity = _make_activity(
            value={"action": "vote", "debate_id": "", "agent": "claude"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 400
        assert "Invalid vote data" in result["body"]["value"]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_missing_agent(self, mock_audit, card_actions, active_debate):
        """Vote with empty agent returns 400."""
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": ""},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_missing_both(self, mock_audit, card_actions):
        """Vote with both empty returns 400."""
        activity = _make_activity(
            value={"action": "vote", "debate_id": "", "agent": ""},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_vote_rbac_denied(self, card_actions, mock_bot, active_debate):
        """Vote with permission denied returns 403."""
        # First call allows card action, second call denies vote
        mock_bot._check_permission.side_effect = [None, {"error": "denied"}]
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "claude"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 403

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_uses_value_field_fallback(self, mock_audit, card_actions, active_debate):
        """Vote reads agent from 'value' field when 'agent' is missing."""
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "value": "gemini"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_creates_debate_entry_in_user_votes(self, mock_audit, card_actions):
        """Vote for a debate not yet in _user_votes creates the entry."""
        from aragora.server.handlers.bots.teams_utils import _user_votes

        activity = _make_activity(
            value={"action": "vote", "debate_id": "new-debate-999", "agent": "claude"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert "new-debate-999" in _user_votes
        assert _user_votes["new-debate-999"]["user-123"] == "claude"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_returns_adaptive_card(self, mock_audit, card_actions, active_debate):
        """Vote response has adaptive card content type."""
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "claude"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["body"]["type"] == "application/vnd.microsoft.card.adaptive"
        assert result["body"]["value"]["type"] == "AdaptiveCard"


# ===========================================================================
# Summary action tests
# ===========================================================================


class TestSummaryAction:
    """Test _handle_summary."""

    @pytest.mark.asyncio
    async def test_summary_active_debate(self, card_actions, active_debate):
        """Summary for active debate returns adaptive card with facts."""
        activity = _make_activity(value={"action": "summary", "debate_id": active_debate})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["type"] == "application/vnd.microsoft.card.adaptive"
        card_body = result["body"]["value"]["body"]
        assert any("Debate Summary" in item.get("text", "") for item in card_body)

    @pytest.mark.asyncio
    async def test_summary_missing_debate(self, card_actions):
        """Summary for non-existent debate returns not found message."""
        activity = _make_activity(value={"action": "summary", "debate_id": "missing-debate"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert "not found" in result["body"]["value"]

    @pytest.mark.asyncio
    async def test_summary_shows_topic(self, card_actions, active_debate):
        """Summary card includes the debate topic."""
        activity = _make_activity(value={"action": "summary", "debate_id": active_debate})
        result = await card_actions.handle_invoke(activity)
        facts = None
        for item in result["body"]["value"]["body"]:
            if item.get("type") == "FactSet":
                facts = item["facts"]
                break
        assert facts is not None
        topic_fact = next(f for f in facts if f["title"] == "Topic")
        assert "microservices" in topic_fact["value"]

    @pytest.mark.asyncio
    async def test_summary_shows_elapsed_time(self, card_actions, active_debate):
        """Summary card includes elapsed time."""
        activity = _make_activity(value={"action": "summary", "debate_id": active_debate})
        result = await card_actions.handle_invoke(activity)
        facts = None
        for item in result["body"]["value"]["body"]:
            if item.get("type") == "FactSet":
                facts = item["facts"]
                break
        elapsed_fact = next(f for f in facts if f["title"] == "Elapsed")
        assert "minutes" in elapsed_fact["value"]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_summary_shows_leading_agent_with_votes(
        self, mock_audit, card_actions, active_debate
    ):
        """Summary shows leading agent when votes exist."""
        from aragora.server.handlers.bots.teams_utils import _user_votes

        _user_votes[active_debate] = {"u1": "claude", "u2": "claude", "u3": "gpt4"}
        activity = _make_activity(value={"action": "summary", "debate_id": active_debate})
        result = await card_actions.handle_invoke(activity)
        facts = None
        for item in result["body"]["value"]["body"]:
            if item.get("type") == "FactSet":
                facts = item["facts"]
                break
        leading = next((f for f in facts if f["title"] == "Leading Agent"), None)
        assert leading is not None
        assert "claude" in leading["value"]

    @pytest.mark.asyncio
    async def test_summary_no_votes(self, card_actions, active_debate):
        """Summary shows 0 votes when none cast."""
        activity = _make_activity(value={"action": "summary", "debate_id": active_debate})
        result = await card_actions.handle_invoke(activity)
        facts = None
        for item in result["body"]["value"]["body"]:
            if item.get("type") == "FactSet":
                facts = item["facts"]
                break
        votes_fact = next(f for f in facts if f["title"] == "Votes Cast")
        assert votes_fact["value"] == "0"

    @pytest.mark.asyncio
    async def test_summary_truncates_debate_id_in_not_found(self, card_actions):
        """Not-found message truncates debate ID to 8 chars."""
        long_id = "abcdef1234567890abcdef1234567890"
        activity = _make_activity(value={"action": "summary", "debate_id": long_id})
        result = await card_actions.handle_invoke(activity)
        assert long_id[:8] in result["body"]["value"]
        assert long_id not in result["body"]["value"]


# ===========================================================================
# View details action tests
# ===========================================================================


class TestViewDetailsAction:
    """Test _handle_view_details."""

    @pytest.mark.asyncio
    async def test_view_details_missing_debate(self, card_actions, mock_bot):
        """View details for missing debate sends not-found reply."""
        activity = _make_activity(value={"action": "view_details", "debate_id": "missing"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["value"] == "Debate not found"
        mock_bot.send_reply.assert_called_once()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "not found" in reply_text

    @pytest.mark.asyncio
    async def test_view_details_active_debate_with_import_error(
        self, card_actions, mock_bot, active_debate
    ):
        """View details falls back to text when card module import fails."""
        import sys

        activity = _make_activity(value={"action": "view_details", "debate_id": active_debate})

        # Force the import inside _handle_view_details to fail
        saved = sys.modules.get("aragora.server.handlers.bots.teams_cards")
        sys.modules["aragora.server.handlers.bots.teams_cards"] = None  # type: ignore
        try:
            result = await card_actions.handle_invoke(activity)
        finally:
            if saved is not None:
                sys.modules["aragora.server.handlers.bots.teams_cards"] = saved
            else:
                sys.modules.pop("aragora.server.handlers.bots.teams_cards", None)

        assert result["status"] == 200
        assert result["body"]["value"] == "Details sent"
        mock_bot.send_reply.assert_called_once()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "Debate Details" in reply_text
        assert "microservices" in reply_text

    @pytest.mark.asyncio
    async def test_view_details_active_debate_with_card_module(
        self, card_actions, mock_bot, active_debate
    ):
        """View details sends progress card when module is available."""
        mock_card = {"type": "AdaptiveCard"}
        with patch(
            "aragora.server.handlers.bots.teams_cards.create_debate_progress_card",
            return_value=mock_card,
            create=True,
        ):
            activity = _make_activity(value={"action": "view_details", "debate_id": active_debate})
            result = await card_actions.handle_invoke(activity)
            assert result["status"] == 200
            assert result["body"]["value"] == "Details sent"


# ===========================================================================
# View report action tests
# ===========================================================================


class TestViewReportAction:
    """Test _handle_view_report."""

    @pytest.mark.asyncio
    async def test_view_report_sends_link(self, card_actions, mock_bot, active_debate):
        """View report sends a reply with a report link."""
        activity = _make_activity(value={"action": "view_report", "debate_id": active_debate})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["value"] == "Report link sent"
        mock_bot.send_reply.assert_called_once()
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "Full Report" in reply_text
        assert active_debate in reply_text

    @pytest.mark.asyncio
    async def test_view_report_includes_debate_id_in_url(self, card_actions, mock_bot):
        """Report URL includes the debate ID."""
        debate_id = "some-specific-id"
        activity = _make_activity(value={"action": "view_report", "debate_id": debate_id})
        await card_actions.handle_invoke(activity)
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert f"https://aragora.ai/debate/{debate_id}" in reply_text


# ===========================================================================
# View rankings action tests
# ===========================================================================


class TestViewRankingsAction:
    """Test _handle_view_rankings."""

    @pytest.mark.asyncio
    async def test_view_rankings_delegates_to_leaderboard(self, card_actions, mock_bot):
        """View rankings delegates to event processor leaderboard command."""
        activity = _make_activity(value={"action": "view_rankings", "period": "monthly"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["value"] == "Rankings sent"
        mock_bot._get_event_processor.return_value._cmd_leaderboard.assert_called_once()

    @pytest.mark.asyncio
    async def test_view_rankings_default_period(self, card_actions, mock_bot):
        """View rankings works with default period."""
        activity = _make_activity(value={"action": "view_rankings"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200


# ===========================================================================
# Watch debate action tests
# ===========================================================================


class TestWatchDebateAction:
    """Test _handle_watch_debate."""

    @pytest.mark.asyncio
    async def test_watch_active_debate(self, card_actions, active_debate):
        """Watch adds user to watchers list."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        activity = _make_activity(value={"action": "watch", "debate_id": active_debate})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert "updates" in result["body"]["value"]
        assert "user-123" in _active_debates[active_debate]["watchers"]

    @pytest.mark.asyncio
    async def test_watch_missing_debate(self, card_actions):
        """Watch for missing debate returns not found."""
        activity = _make_activity(value={"action": "watch", "debate_id": "missing"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert "not found" in result["body"]["value"].lower()

    @pytest.mark.asyncio
    async def test_watch_already_watching(self, card_actions, active_debate):
        """Watching twice does not duplicate user in watchers."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        activity = _make_activity(value={"action": "watch", "debate_id": active_debate})
        await card_actions.handle_invoke(activity)
        await card_actions.handle_invoke(activity)
        watchers = _active_debates[active_debate]["watchers"]
        assert watchers.count("user-123") == 1

    @pytest.mark.asyncio
    async def test_watch_multiple_users(self, card_actions, active_debate):
        """Multiple users can watch the same debate."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        a1 = _make_activity(user_id="user-A", value={"action": "watch", "debate_id": active_debate})
        a2 = _make_activity(user_id="user-B", value={"action": "watch", "debate_id": active_debate})
        await card_actions.handle_invoke(a1)
        await card_actions.handle_invoke(a2)
        watchers = _active_debates[active_debate]["watchers"]
        assert "user-A" in watchers
        assert "user-B" in watchers


# ===========================================================================
# Share result action tests
# ===========================================================================


class TestShareResultAction:
    """Test _handle_share_result."""

    @pytest.mark.asyncio
    async def test_share_active_debate(self, card_actions, mock_bot, active_debate):
        """Share sends consensus card for active debate."""
        activity = _make_activity(value={"action": "share", "debate_id": active_debate})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["value"] == "Result shared"
        mock_bot.send_card.assert_called_once()

    @pytest.mark.asyncio
    async def test_share_missing_debate(self, card_actions):
        """Share for missing debate returns not available."""
        activity = _make_activity(value={"action": "share", "debate_id": "missing"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert "not available" in result["body"]["value"]

    @pytest.mark.asyncio
    async def test_share_sends_correct_topic(self, card_actions, mock_bot, active_debate):
        """Share card fallback text includes the topic."""
        activity = _make_activity(value={"action": "share", "debate_id": active_debate})
        await card_actions.handle_invoke(activity)
        fallback_text = mock_bot.send_card.call_args[0][2]
        assert "microservices" in fallback_text


# ===========================================================================
# Start debate prompt action tests
# ===========================================================================


class TestStartDebatePromptAction:
    """Test _handle_start_debate_prompt."""

    @pytest.mark.asyncio
    async def test_start_debate_prompt_returns_task_module(self, card_actions):
        """Start debate prompt returns a task module card."""
        activity = _make_activity(value={"action": "start_debate_prompt"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert "task" in result["body"]
        task = result["body"]["task"]
        assert task["type"] == "continue"
        assert task["value"]["title"] == "Start a Debate"

    @pytest.mark.asyncio
    async def test_start_debate_prompt_has_input_field(self, card_actions):
        """Task module card has a text input for the topic."""
        activity = _make_activity(value={"action": "start_debate_prompt"})
        result = await card_actions.handle_invoke(activity)
        card = result["body"]["task"]["value"]["card"]["content"]
        inputs = [b for b in card["body"] if b.get("type") == "Input.Text"]
        assert len(inputs) == 1
        assert inputs[0]["id"] == "debate_topic"

    @pytest.mark.asyncio
    async def test_start_debate_prompt_has_submit_action(self, card_actions):
        """Task module card has a submit action button."""
        activity = _make_activity(value={"action": "start_debate_prompt"})
        result = await card_actions.handle_invoke(activity)
        card = result["body"]["task"]["value"]["card"]["content"]
        actions = card["actions"]
        assert len(actions) == 1
        assert actions[0]["data"]["action"] == "start_debate_from_task_module"


# ===========================================================================
# Compose extension submit tests
# ===========================================================================


class TestComposeExtensionSubmit:
    """Test _handle_compose_extension_submit."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    async def test_start_debate_success(self, mock_build_card, mock_start, card_actions):
        """Successful debate start via compose extension."""
        mock_start.return_value = "debate-new-123"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "startDebate", "data": {"topic": "AI Ethics"}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        ext = result["body"]["composeExtension"]
        assert ext["type"] == "result"
        assert len(ext["attachments"]) == 1

    @pytest.mark.asyncio
    async def test_start_debate_missing_topic(self, card_actions):
        """Start debate with empty topic returns error message."""
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "startDebate", "data": {}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert "provide a debate topic" in result["body"]["composeExtension"]["text"]

    @pytest.mark.asyncio
    async def test_start_debate_rbac_denied(self, card_actions, mock_bot):
        """Start debate with RBAC denied returns permission error."""
        mock_bot._check_permission.return_value = {"error": "denied"}
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "startDebate", "data": {"topic": "Test"}},
        )
        result = await card_actions.handle_invoke(activity)
        assert "permission" in result["body"]["composeExtension"]["text"].lower()

    @pytest.mark.asyncio
    async def test_unknown_command(self, card_actions):
        """Unknown command returns 'not recognized' message."""
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "unknownCmd", "data": {}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert "not recognized" in result["body"]["composeExtension"]["text"]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    async def test_start_debate_uses_debate_topic_field(
        self, mock_build_card, mock_start, card_actions
    ):
        """startDebate reads from data.debate_topic as fallback."""
        mock_start.return_value = "d-001"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "startDebate", "data": {"debate_topic": "Fallback topic"}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        mock_start.assert_called_once()
        assert mock_start.call_args[1]["topic"] == "Fallback topic"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    async def test_start_debate_preview_truncates_topic(
        self, mock_build_card, mock_start, card_actions
    ):
        """Preview title truncates long topics to 50 chars."""
        mock_start.return_value = "d-002"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        long_topic = "A" * 100
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "startDebate", "data": {"topic": long_topic}},
        )
        result = await card_actions.handle_invoke(activity)
        preview = result["body"]["composeExtension"]["attachments"][0]["preview"]
        assert len(preview["content"]["title"]) <= len("Debate: ") + 50

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    async def test_start_debate_passes_attachments(self, mock_build_card, mock_start, card_actions):
        """Start debate passes activity attachments."""
        mock_start.return_value = "d-003"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        attachments = [{"contentType": "text/plain", "content": "extra"}]
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "startDebate", "data": {"topic": "Test"}},
            attachments=attachments,
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert mock_start.call_args[1]["attachments"] == attachments

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    async def test_start_debate_non_list_attachments(
        self, mock_build_card, mock_start, card_actions
    ):
        """Non-list attachments are replaced with empty list."""
        mock_start.return_value = "d-004"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "startDebate", "data": {"topic": "Test"}},
        )
        activity["attachments"] = "not-a-list"
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert mock_start.call_args[1]["attachments"] == []


# ===========================================================================
# Compose extension query tests
# ===========================================================================


class TestComposeExtensionQuery:
    """Test _handle_compose_extension_query."""

    @pytest.mark.asyncio
    async def test_query_no_debates(self, card_actions):
        """Query with no active debates returns empty results."""
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": [{"name": "query", "value": "test"}]},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["composeExtension"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_query_matches_topic(self, card_actions, active_debate):
        """Query matching topic returns the debate."""
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": [{"name": "query", "value": "microservices"}]},
        )
        result = await card_actions.handle_invoke(activity)
        attachments = result["body"]["composeExtension"]["attachments"]
        assert len(attachments) == 1

    @pytest.mark.asyncio
    async def test_query_no_match(self, card_actions, active_debate):
        """Query not matching any topic returns empty results."""
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": [{"name": "query", "value": "quantum_physics_xyz"}]},
        )
        result = await card_actions.handle_invoke(activity)
        attachments = result["body"]["composeExtension"]["attachments"]
        assert len(attachments) == 0

    @pytest.mark.asyncio
    async def test_query_empty_returns_all(self, card_actions, active_debate):
        """Empty query returns all debates."""
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": [{"name": "query", "value": ""}]},
        )
        result = await card_actions.handle_invoke(activity)
        attachments = result["body"]["composeExtension"]["attachments"]
        assert len(attachments) == 1

    @pytest.mark.asyncio
    async def test_query_no_parameters(self, card_actions, active_debate):
        """Query with no parameters returns all debates."""
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": []},
        )
        result = await card_actions.handle_invoke(activity)
        attachments = result["body"]["composeExtension"]["attachments"]
        assert len(attachments) == 1

    @pytest.mark.asyncio
    async def test_query_case_insensitive(self, card_actions, active_debate):
        """Query is case-insensitive."""
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": [{"name": "query", "value": "MICROSERVICES"}]},
        )
        result = await card_actions.handle_invoke(activity)
        attachments = result["body"]["composeExtension"]["attachments"]
        assert len(attachments) == 1

    @pytest.mark.asyncio
    async def test_query_limit_10(self, card_actions):
        """Query returns at most 10 results."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        for i in range(15):
            _active_debates[f"debate-{i}"] = {"topic": f"Topic {i}", "started_at": time.time()}
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": []},
        )
        result = await card_actions.handle_invoke(activity)
        attachments = result["body"]["composeExtension"]["attachments"]
        assert len(attachments) == 10

    @pytest.mark.asyncio
    async def test_query_rbac_denied(self, card_actions, mock_bot):
        """Query with RBAC denied returns permission message."""
        mock_bot._check_permission.return_value = {"error": "denied"}
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": []},
        )
        result = await card_actions.handle_invoke(activity)
        assert "permission" in result["body"]["composeExtension"]["text"].lower()

    @pytest.mark.asyncio
    async def test_query_result_has_preview(self, card_actions, active_debate):
        """Query results include a thumbnail preview."""
        activity = _make_activity(
            invoke_name="composeExtension/query",
            value={"parameters": []},
        )
        result = await card_actions.handle_invoke(activity)
        attachment = result["body"]["composeExtension"]["attachments"][0]
        assert "preview" in attachment
        assert attachment["preview"]["contentType"] == "application/vnd.microsoft.card.thumbnail"


# ===========================================================================
# Task module fetch tests
# ===========================================================================


class TestTaskModuleFetch:
    """Test _handle_task_module_fetch."""

    @pytest.mark.asyncio
    async def test_fetch_start_debate_command(self, card_actions):
        """Fetch with startDebate commandId returns debate prompt."""
        activity = _make_activity(
            invoke_name="composeExtension/fetchTask",
            value={"commandId": "startDebate"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["task"]["value"]["title"] == "Start a Debate"

    @pytest.mark.asyncio
    async def test_fetch_start_debate_from_data_action(self, card_actions):
        """Fetch with start_debate_prompt action in data returns debate prompt."""
        activity = _make_activity(
            invoke_name="task/fetch",
            value={"data": {"action": "start_debate_prompt"}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["task"]["value"]["title"] == "Start a Debate"

    @pytest.mark.asyncio
    async def test_fetch_start_debate_from_nested_command_id(self, card_actions):
        """Fetch reads commandId from data when not at top level."""
        activity = _make_activity(
            invoke_name="task/fetch",
            value={"data": {"commandId": "startDebate"}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["task"]["value"]["title"] == "Start a Debate"

    @pytest.mark.asyncio
    async def test_fetch_generic_returns_usage_message(self, card_actions):
        """Fetch without specific command returns generic usage card."""
        activity = _make_activity(
            invoke_name="task/fetch",
            value={"commandId": "other"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        task = result["body"]["task"]
        assert task["type"] == "continue"
        assert task["value"]["title"] == "Aragora"
        card_body = task["value"]["card"]["content"]["body"]
        assert any("@Aragora" in item.get("text", "") for item in card_body)

    @pytest.mark.asyncio
    async def test_fetch_rbac_denied(self, card_actions, mock_bot):
        """Fetch with RBAC denied returns permission message."""
        mock_bot._check_permission.return_value = {"error": "denied"}
        activity = _make_activity(
            invoke_name="task/fetch",
            value={},
        )
        result = await card_actions.handle_invoke(activity)
        assert "permission" in result["body"]["task"]["value"].lower()


# ===========================================================================
# Task module submit tests
# ===========================================================================


class TestTaskModuleSubmit:
    """Test _handle_task_module_submit."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_submit_start_debate(
        self, mock_audit, mock_build_card, mock_start, card_actions, mock_bot
    ):
        """Submit with start_debate_from_task_module starts debate."""
        mock_start.return_value = "debate-tm-001"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity(
            invoke_name="task/submit",
            value={
                "data": {"action": "start_debate_from_task_module", "debate_topic": "ML Ethics"}
            },
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        mock_start.assert_called_once()
        mock_bot.send_card.assert_called_once()
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_submit_start_debate_audit_fields(
        self, mock_audit, mock_build_card, mock_start, card_actions, mock_bot
    ):
        """Submit audit log includes correct fields."""
        mock_start.return_value = "debate-tm-002"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity(
            invoke_name="task/submit",
            value={
                "data": {"action": "start_debate_from_task_module", "debate_topic": "Topic XYZ"}
            },
        )
        await card_actions.handle_invoke(activity)
        mock_audit.assert_called_once_with(
            user_id="teams:user-123",
            resource_type="debate",
            resource_id="debate-tm-002",
            action="create",
            platform="teams",
            task_preview="Topic XYZ",
        )

    @pytest.mark.asyncio
    async def test_submit_empty_topic(self, card_actions, mock_bot):
        """Submit with empty topic returns Done without starting debate."""
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {"action": "start_debate_from_task_module", "debate_topic": ""}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["task"]["value"] == "Done"
        mock_bot.send_card.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_no_action(self, card_actions):
        """Submit with no action returns Done."""
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["task"]["value"] == "Done"

    @pytest.mark.asyncio
    async def test_submit_unknown_action(self, card_actions):
        """Submit with unknown action returns Done."""
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {"action": "unknown_action"}},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_submit_rbac_denied(self, card_actions, mock_bot):
        """Submit with RBAC denied returns permission message."""
        mock_bot._check_permission.return_value = {"error": "denied"}
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {"action": "start_debate_from_task_module", "debate_topic": "Test"}},
        )
        result = await card_actions.handle_invoke(activity)
        assert "permission" in result["body"]["task"]["value"].lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_submit_passes_conversation_and_service_url(
        self, mock_audit, mock_build_card, mock_start, card_actions, mock_bot
    ):
        """Submit passes correct conversation_id and service_url to debate starter."""
        mock_start.return_value = "d-005"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {"action": "start_debate_from_task_module", "debate_topic": "Topic"}},
            conversation_id="my-conv",
            service_url="https://my.service/",
        )
        await card_actions.handle_invoke(activity)
        assert mock_start.call_args[1]["conversation_id"] == "my-conv"
        assert mock_start.call_args[1]["service_url"] == "https://my.service/"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_submit_non_list_attachments_replaced(
        self, mock_audit, mock_build_card, mock_start, card_actions, mock_bot
    ):
        """Non-list attachments are replaced with empty list in submit."""
        mock_start.return_value = "d-006"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {"action": "start_debate_from_task_module", "debate_topic": "Topic"}},
        )
        activity["attachments"] = 42  # Not a list
        await card_actions.handle_invoke(activity)
        assert mock_start.call_args[1]["attachments"] == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_submit_build_card_no_vote_buttons(
        self, mock_audit, mock_build_card, mock_start, card_actions, mock_bot
    ):
        """Submit builds debate card with include_vote_buttons=False."""
        mock_start.return_value = "d-007"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {"action": "start_debate_from_task_module", "debate_topic": "Topic"}},
        )
        await card_actions.handle_invoke(activity)
        assert mock_build_card.call_args[1]["include_vote_buttons"] is False


# ===========================================================================
# Link unfurling tests
# ===========================================================================


class TestLinkUnfurling:
    """Test _handle_link_unfurling."""

    @pytest.mark.asyncio
    async def test_unfurl_debate_url_found(self, card_actions, active_debate):
        """Unfurling a debate URL with active debate returns card."""
        activity = _make_activity(
            invoke_name="composeExtension/queryLink",
            value={"url": f"https://aragora.ai/debate/{active_debate}"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        attachments = result["body"]["composeExtension"]["attachments"]
        assert len(attachments) == 1

    @pytest.mark.asyncio
    async def test_unfurl_debate_url_not_found(self, card_actions):
        """Unfurling a debate URL with no active debate returns empty."""
        activity = _make_activity(
            invoke_name="composeExtension/queryLink",
            value={"url": "https://aragora.ai/debate/abc12345-dead-beef-1234-567890abcdef"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["composeExtension"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_unfurl_non_debate_url(self, card_actions):
        """Unfurling a non-debate URL returns empty attachments."""
        activity = _make_activity(
            invoke_name="composeExtension/queryLink",
            value={"url": "https://aragora.ai/settings"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["composeExtension"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_unfurl_empty_url(self, card_actions):
        """Unfurling with empty URL returns empty attachments."""
        activity = _make_activity(
            invoke_name="composeExtension/queryLink",
            value={"url": ""},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["composeExtension"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_unfurl_rbac_denied(self, card_actions, mock_bot):
        """Unfurling with RBAC denied returns empty attachments silently."""
        mock_bot._check_permission.return_value = {"error": "denied"}
        activity = _make_activity(
            invoke_name="composeExtension/queryLink",
            value={"url": "https://aragora.ai/debate/some-id"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["composeExtension"]["attachments"] == []

    @pytest.mark.asyncio
    async def test_unfurl_preview_truncates_topic(self, card_actions, active_debate):
        """Unfurl preview truncates topic to 50 chars."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        _active_debates[active_debate]["topic"] = "X" * 200
        activity = _make_activity(
            invoke_name="composeExtension/queryLink",
            value={"url": f"https://aragora.ai/debate/{active_debate}"},
        )
        result = await card_actions.handle_invoke(activity)
        attachment = result["body"]["composeExtension"]["attachments"][0]
        title = attachment["preview"]["content"]["title"]
        assert len(title) <= len("Debate: ") + 50

    @pytest.mark.asyncio
    async def test_unfurl_preview_shows_debate_id(self, card_actions, active_debate):
        """Unfurl preview text shows truncated debate ID."""
        activity = _make_activity(
            invoke_name="composeExtension/queryLink",
            value={"url": f"https://aragora.ai/debate/{active_debate}"},
        )
        result = await card_actions.handle_invoke(activity)
        attachment = result["body"]["composeExtension"]["attachments"][0]
        preview_text = attachment["preview"]["content"]["text"]
        assert active_debate[:8] in preview_text


# ===========================================================================
# RBAC permission edge cases
# ===========================================================================


class TestRBACEdgeCases:
    """Test RBAC permission edge cases across multiple actions."""

    @pytest.mark.asyncio
    async def test_card_action_rbac_returns_adaptive_card_error(self, card_actions, mock_bot):
        """RBAC denial on card action returns proper adaptive card body."""
        mock_bot._check_permission.return_value = {"error": "denied"}
        activity = _make_activity(value={"action": "summary", "debate_id": "d1"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 403
        card = result["body"]["value"]
        assert card["type"] == "AdaptiveCard"
        assert card["version"] == "1.4"

    @pytest.mark.asyncio
    async def test_vote_rbac_returns_separate_card(self, card_actions, mock_bot, active_debate):
        """Vote RBAC denial returns its own card mentioning voting."""
        mock_bot._check_permission.side_effect = [None, {"error": "denied"}]
        activity = _make_activity(
            value={"action": "vote", "debate_id": active_debate, "agent": "claude"},
        )
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 403
        body_items = result["body"]["value"]["body"]
        texts = [item.get("text", "") for item in body_items]
        assert any("vote" in t.lower() for t in texts)

    @pytest.mark.asyncio
    async def test_compose_submit_rbac_only_for_start_debate(self, card_actions, mock_bot):
        """Compose submit only checks RBAC for startDebate command."""
        mock_bot._check_permission.return_value = {"error": "denied"}
        # Non-startDebate command should not check RBAC
        activity = _make_activity(
            invoke_name="composeExtension/submitAction",
            value={"commandId": "otherCommand"},
        )
        result = await card_actions.handle_invoke(activity)
        # Should not get permission error for non-startDebate
        assert "not recognized" in result["body"]["composeExtension"]["text"]


# ===========================================================================
# AGENT_DISPLAY_NAMES tests
# ===========================================================================


class TestAgentDisplayNames:
    """Test the AGENT_DISPLAY_NAMES constant."""

    def test_all_expected_agents(self):
        """AGENT_DISPLAY_NAMES contains all expected agent mappings."""
        from aragora.server.handlers.bots.teams.cards import AGENT_DISPLAY_NAMES

        expected = {
            "claude",
            "gpt4",
            "gemini",
            "mistral",
            "deepseek",
            "grok",
            "qwen",
            "kimi",
            "anthropic-api",
            "openai-api",
        }
        assert set(AGENT_DISPLAY_NAMES.keys()) == expected

    def test_display_names_are_strings(self):
        """All display names are non-empty strings."""
        from aragora.server.handlers.bots.teams.cards import AGENT_DISPLAY_NAMES

        for key, value in AGENT_DISPLAY_NAMES.items():
            assert isinstance(value, str)
            assert len(value) > 0


# ===========================================================================
# Permission constants tests
# ===========================================================================


class TestPermissionConstants:
    """Test permission constant values."""

    def test_permission_constants_defined(self):
        """Permission constants are defined with correct values."""
        from aragora.server.handlers.bots.teams.cards import (
            PERM_TEAMS_CARDS_RESPOND,
            PERM_TEAMS_DEBATES_CREATE,
            PERM_TEAMS_DEBATES_VOTE,
        )

        assert PERM_TEAMS_CARDS_RESPOND == "teams:cards:respond"
        assert PERM_TEAMS_DEBATES_CREATE == "teams:debates:create"
        assert PERM_TEAMS_DEBATES_VOTE == "teams:debates:vote"


# ===========================================================================
# Edge cases and misc tests
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    @pytest.mark.asyncio
    async def test_handle_invoke_empty_activity(self, card_actions):
        """handle_invoke with minimal activity works."""
        activity = {"from": {}, "value": {}}
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_handle_invoke_missing_from(self, card_actions):
        """handle_invoke with no 'from' field uses empty user_id."""
        activity = {"name": "", "value": {"action": ""}}
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_handle_invoke_missing_value(self, card_actions):
        """handle_invoke with no 'value' field uses empty dict."""
        activity = {"name": "adaptiveCard/action", "from": {"id": "u1"}}
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_summary_long_topic_truncated(self, card_actions):
        """Summary truncates topic to 100 chars in facts."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        long_topic = "Z" * 500
        _active_debates["long-topic-debate"] = {
            "topic": long_topic,
            "started_at": time.time(),
        }
        activity = _make_activity(value={"action": "summary", "debate_id": "long-topic-debate"})
        result = await card_actions.handle_invoke(activity)
        facts = None
        for item in result["body"]["value"]["body"]:
            if item.get("type") == "FactSet":
                facts = item["facts"]
                break
        topic_fact = next(f for f in facts if f["title"] == "Topic")
        assert len(topic_fact["value"]) <= 100

    @pytest.mark.asyncio
    async def test_summary_no_started_at(self, card_actions):
        """Summary handles missing started_at gracefully."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        _active_debates["no-time"] = {"topic": "Test"}
        activity = _make_activity(value={"action": "summary", "debate_id": "no-time"})
        result = await card_actions.handle_invoke(activity)
        assert result["status"] == 200
        facts = None
        for item in result["body"]["value"]["body"]:
            if item.get("type") == "FactSet":
                facts = item["facts"]
                break
        elapsed_fact = next(f for f in facts if f["title"] == "Elapsed")
        assert "0.0 minutes" in elapsed_fact["value"]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_vote_multiple_users(self, mock_audit, card_actions, active_debate):
        """Multiple users can vote on the same debate."""
        from aragora.server.handlers.bots.teams_utils import _user_votes

        a1 = _make_activity(
            user_id="user-A",
            value={"action": "vote", "debate_id": active_debate, "agent": "claude"},
        )
        a2 = _make_activity(
            user_id="user-B",
            value={"action": "vote", "debate_id": active_debate, "agent": "gpt4"},
        )
        await card_actions.handle_invoke(a1)
        await card_actions.handle_invoke(a2)
        assert _user_votes[active_debate]["user-A"] == "claude"
        assert _user_votes[active_debate]["user-B"] == "gpt4"

    @pytest.mark.asyncio
    async def test_view_details_debate_with_defaults(self, card_actions, mock_bot):
        """View details uses defaults for missing round/phase info."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        _active_debates["min-debate"] = {
            "topic": "Minimal debate",
            "started_at": time.time(),
        }

        # Force ImportError for the progress card
        import sys

        saved = sys.modules.get("aragora.server.handlers.bots.teams_cards")
        sys.modules["aragora.server.handlers.bots.teams_cards"] = None  # type: ignore
        try:
            activity = _make_activity(value={"action": "view_details", "debate_id": "min-debate"})
            result = await card_actions.handle_invoke(activity)
        finally:
            if saved is not None:
                sys.modules["aragora.server.handlers.bots.teams_cards"] = saved
            else:
                sys.modules.pop("aragora.server.handlers.bots.teams_cards", None)

        assert result["status"] == 200
        reply_text = mock_bot.send_reply.call_args[0][1]
        assert "**Round:** 1/" in reply_text

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.bots.teams.cards._start_teams_debate", new_callable=AsyncMock)
    @patch("aragora.server.handlers.bots.teams.cards.build_debate_card")
    @patch("aragora.server.handlers.bots.teams.cards.audit_data")
    async def test_submit_task_preview_truncated(
        self, mock_audit, mock_build_card, mock_start, card_actions, mock_bot
    ):
        """Audit log truncates task_preview to 100 chars."""
        mock_start.return_value = "d-long"
        mock_build_card.return_value = {"type": "AdaptiveCard"}
        long_topic = "W" * 200
        activity = _make_activity(
            invoke_name="task/submit",
            value={"data": {"action": "start_debate_from_task_module", "debate_topic": long_topic}},
        )
        await card_actions.handle_invoke(activity)
        audit_kwargs = mock_audit.call_args[1]
        assert len(audit_kwargs["task_preview"]) == 100

    @pytest.mark.asyncio
    async def test_card_actions_stores_bot_reference(self, mock_bot):
        """TeamsCardActions stores reference to bot."""
        from aragora.server.handlers.bots.teams.cards import TeamsCardActions

        ca = TeamsCardActions(mock_bot)
        assert ca.bot is mock_bot


# ===========================================================================
# Module exports test
# ===========================================================================


class TestModuleExports:
    """Test __all__ exports."""

    def test_all_exports(self):
        """Module __all__ contains expected exports."""
        from aragora.server.handlers.bots.teams import cards

        assert "TeamsCardActions" in cards.__all__
        assert "AGENT_DISPLAY_NAMES" in cards.__all__
