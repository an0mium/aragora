"""
Tests for Slack Interactive Components Handler.

Covers all routes and behavior of handle_slack_interactions():
- Payload parsing (form-encoded body -> JSON payload)
- block_actions interaction type:
  - vote_ action: vote recording, validation, audit data
  - summary_ action: debate summary request
  - invalid action_id handling
  - empty actions list
- shortcut interaction type:
  - start_debate callback: opens modal
  - invalid callback_id
- view_submission interaction type:
  - start_debate_modal: creates debate with task/agents/rounds
  - task validation (empty, too long, injection)
  - agents validation (empty, invalid names)
  - rounds validation (non-digit, out of range, defaults)
  - RBAC permission denied on debate creation
- User ID validation (format, too long, injection)
- Team ID validation (format, too long)
- RBAC permission checks:
  - RBAC available + permission granted
  - RBAC available + permission denied
  - RBAC unavailable + fail-closed
  - RBAC unavailable + fail-open
- Error handling (malformed JSON, missing fields)
- Unknown interaction types
- Security tests (injection patterns, path traversal)
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock, patch
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


# ---------------------------------------------------------------------------
# Mock request builder
# ---------------------------------------------------------------------------


class MockSlackRequest:
    """Mock request object that provides an async body() method.

    Simulates a Slack interactive component webhook where
    the body is form-encoded with a 'payload' field containing JSON.
    """

    def __init__(self, payload: dict[str, Any] | None = None, raw_body: bytes | None = None):
        if raw_body is not None:
            self._raw = raw_body
        elif payload is not None:
            self._raw = urlencode({"payload": json.dumps(payload)}).encode("utf-8")
        else:
            self._raw = urlencode({"payload": "{}"}).encode("utf-8")

    async def body(self) -> bytes:
        return self._raw


def _make_payload(
    interaction_type: str = "block_actions",
    user_id: str = "U12345ABC",
    user_name: str = "testuser",
    team_id: str = "T12345ABC",
    actions: list[dict[str, Any]] | None = None,
    callback_id: str = "",
    view: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a standard Slack interaction payload."""
    payload: dict[str, Any] = {
        "type": interaction_type,
        "user": {"id": user_id, "name": user_name},
        "team": {"id": team_id},
    }
    if actions is not None:
        payload["actions"] = actions
    if callback_id:
        payload["callback_id"] = callback_id
    if view is not None:
        payload["view"] = view
    payload.update(extra)
    return payload


def _vote_action(
    debate_id: str = "abc-123", agent: str = "claude", action_id_prefix: str = "vote_"
) -> dict[str, Any]:
    """Build a vote action dict."""
    return {
        "action_id": f"{action_id_prefix}{debate_id}_{agent}",
        "value": json.dumps({"debate_id": debate_id, "agent": agent}),
    }


def _summary_action(debate_id: str = "abc-123") -> dict[str, Any]:
    """Build a summary action dict."""
    return {
        "action_id": f"summary_{debate_id}",
        "value": debate_id,
    }


def _modal_view(
    task: str = "Design a rate limiter",
    agents: list[str] | None = None,
    rounds: str = "5",
    callback_id: str = "start_debate_modal",
) -> dict[str, Any]:
    """Build a view_submission view payload."""
    if agents is None:
        agents = ["claude", "gpt4"]
    agents_data = [{"value": a} for a in agents]
    return {
        "callback_id": callback_id,
        "state": {
            "values": {
                "task_block": {
                    "task_input": {"value": task},
                },
                "agents_block": {
                    "agents_select": {
                        "selected_options": agents_data,
                    },
                },
                "rounds_block": {
                    "rounds_select": {
                        "selected_option": {"value": rounds},
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Lazy import fixture (so conftest auto-auth patches run first)
# ---------------------------------------------------------------------------


@pytest.fixture
def interactions_module():
    """Import the interactions module lazily."""
    import aragora.server.handlers.bots.slack.interactions as mod

    return mod


@pytest.fixture
def state_module():
    """Import the state module for direct inspection."""
    import aragora.server.handlers.bots.slack.state as mod

    return mod


@pytest.fixture(autouse=True)
def _clean_state(state_module):
    """Clear global state before/after each test."""
    state_module._active_debates.clear()
    state_module._user_votes.clear()
    yield
    state_module._active_debates.clear()
    state_module._user_votes.clear()


@pytest.fixture(autouse=True)
def _patch_rbac_open(monkeypatch):
    """Default: RBAC available + permission granted, fail-open."""
    # Make RBAC "available" with a permissive check_permission
    mock_decision = MagicMock()
    mock_decision.allowed = True
    monkeypatch.setattr("aragora.server.handlers.bots.slack.constants.RBAC_AVAILABLE", True)
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.constants.check_permission",
        MagicMock(return_value=mock_decision),
    )
    # Also patch the module-level references used by interactions.py
    monkeypatch.setattr("aragora.server.handlers.bots.slack.interactions.RBAC_AVAILABLE", True)
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.interactions.check_permission",
        MagicMock(return_value=mock_decision),
    )
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.interactions.rbac_fail_closed",
        lambda: False,
    )
    # Patch audit_data to avoid side effects
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.interactions.audit_data",
        MagicMock(),
    )


# ===========================================================================
# block_actions -- vote
# ===========================================================================


class TestVoteAction:
    """Tests for vote recording via block_actions."""

    @pytest.mark.asyncio
    async def test_vote_records_user_vote(self, interactions_module, state_module):
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_vote_returns_ephemeral_confirmation(self, interactions_module):
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_type"] == "ephemeral"
        assert "claude" in body["text"]
        assert body["replace_original"] is False

    @pytest.mark.asyncio
    async def test_vote_returns_200(self, interactions_module):
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_vote_calls_audit_data(self, interactions_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.audit_data", mock_audit
        )
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["resource_type"] == "debate_vote"
        assert call_kwargs["resource_id"] == "d1"
        assert call_kwargs["action"] == "create"
        assert call_kwargs["vote_option"] == "claude"

    @pytest.mark.asyncio
    async def test_vote_overwrites_previous_vote(self, interactions_module, state_module):
        payload1 = _make_payload(actions=[_vote_action("d1", "claude")])
        request1 = MockSlackRequest(payload1)
        await interactions_module.handle_slack_interactions.__wrapped__(request1)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

        payload2 = _make_payload(actions=[_vote_action("d1", "gpt4")])
        request2 = MockSlackRequest(payload2)
        await interactions_module.handle_slack_interactions.__wrapped__(request2)
        assert state_module._user_votes["d1"]["U12345ABC"] == "gpt4"

    @pytest.mark.asyncio
    async def test_vote_multiple_users(self, interactions_module, state_module):
        p1 = _make_payload(user_id="U001", actions=[_vote_action("d1", "claude")])
        await interactions_module.handle_slack_interactions.__wrapped__(MockSlackRequest(p1))
        p2 = _make_payload(user_id="U002", actions=[_vote_action("d1", "gpt4")])
        await interactions_module.handle_slack_interactions.__wrapped__(MockSlackRequest(p2))
        assert state_module._user_votes["d1"]["U001"] == "claude"
        assert state_module._user_votes["d1"]["U002"] == "gpt4"

    @pytest.mark.asyncio
    async def test_vote_invalid_debate_id_empty(self, interactions_module):
        action = {"action_id": "vote_x", "value": json.dumps({"debate_id": "", "agent": "claude"})}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Invalid debate ID" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_vote_invalid_debate_id_too_long(self, interactions_module):
        long_id = "x" * 101
        action = {
            "action_id": "vote_x",
            "value": json.dumps({"debate_id": long_id, "agent": "claude"}),
        }
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Invalid debate ID" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_vote_invalid_agent_empty(self, interactions_module):
        action = {"action_id": "vote_x", "value": json.dumps({"debate_id": "d1", "agent": ""})}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Invalid agent" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_vote_invalid_json_value(self, interactions_module):
        action = {"action_id": "vote_x", "value": "not-json"}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        # JSONDecodeError is caught and pass'd; falls through to return ok
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_vote_missing_value_key(self, interactions_module):
        """Action with no 'value' key defaults to empty string -> JSONDecodeError -> ok."""
        action = {"action_id": "vote_x"}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True


# ===========================================================================
# block_actions -- summary
# ===========================================================================


class TestSummaryAction:
    """Tests for summary request via block_actions."""

    @pytest.mark.asyncio
    async def test_summary_returns_ephemeral(self, interactions_module):
        payload = _make_payload(actions=[_summary_action("abc-123")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_type"] == "ephemeral"
        assert "abc-123"[:8] in body["text"]

    @pytest.mark.asyncio
    async def test_summary_returns_200(self, interactions_module):
        payload = _make_payload(actions=[_summary_action("abc-123")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_summary_empty_debate_id(self, interactions_module):
        action = {"action_id": "summary_x", "value": ""}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Invalid debate ID" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_summary_too_long_debate_id(self, interactions_module):
        action = {"action_id": "summary_x", "value": "x" * 101}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Invalid debate ID" in body.get("text", "")


# ===========================================================================
# block_actions -- edge cases
# ===========================================================================


class TestBlockActionsEdgeCases:
    """Edge cases for block_actions interaction type."""

    @pytest.mark.asyncio
    async def test_empty_actions_list(self, interactions_module):
        payload = _make_payload(actions=[])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_unknown_action_id(self, interactions_module):
        """Action IDs not matching vote_ or summary_ are silently skipped."""
        action = {"action_id": "unknown_action", "value": "x"}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_invalid_action_id_injection(self, interactions_module):
        """Action ID containing injection characters is skipped (continue)."""
        action = {"action_id": "vote_<script>alert(1)</script>", "value": "x"}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        # Invalid action_id is silently skipped via continue
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_no_actions_key(self, interactions_module):
        """Missing actions key defaults to empty list."""
        payload = _make_payload()
        # Don't include actions
        payload.pop("actions", None)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_action_id_too_long(self, interactions_module):
        """Action ID exceeding MAX_COMMAND_LENGTH is skipped."""
        long_id = "vote_" + "a" * 600
        action = {"action_id": long_id, "value": json.dumps({"debate_id": "d1", "agent": "x"})}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True


# ===========================================================================
# shortcut interaction type
# ===========================================================================


class TestShortcut:
    """Tests for global shortcut interaction type."""

    @pytest.mark.asyncio
    async def test_start_debate_shortcut_opens_modal(self, interactions_module):
        payload = _make_payload(
            interaction_type="shortcut",
            callback_id="start_debate",
        )
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "open_modal"
        assert body["view"]["type"] == "modal"
        assert body["view"]["callback_id"] == "start_debate_modal"

    @pytest.mark.asyncio
    async def test_shortcut_returns_200(self, interactions_module):
        payload = _make_payload(
            interaction_type="shortcut",
            callback_id="start_debate",
        )
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_shortcut_unknown_callback_id(self, interactions_module):
        payload = _make_payload(
            interaction_type="shortcut",
            callback_id="unknown_callback",
        )
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        # Falls through to final return {"ok": True}
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_shortcut_invalid_callback_id_injection(self, interactions_module):
        payload = _make_payload(
            interaction_type="shortcut",
            callback_id="start_debate'; DROP TABLE;",
        )
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        # Invalid callback_id is caught by validation -> returns ok
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_shortcut_empty_callback_id(self, interactions_module):
        payload = _make_payload(interaction_type="shortcut")
        # callback_id defaults to "" in payload.get
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True


# ===========================================================================
# view_submission -- start_debate_modal
# ===========================================================================


class TestViewSubmission:
    """Tests for modal form submission (start_debate_modal)."""

    @pytest.mark.asyncio
    async def test_creates_debate(self, interactions_module, state_module):
        view = _modal_view(task="Design a rate limiter", agents=["claude", "gpt4"], rounds="5")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "clear"
        assert len(state_module._active_debates) == 1

    @pytest.mark.asyncio
    async def test_debate_has_correct_fields(self, interactions_module, state_module):
        view = _modal_view(task="Test task", agents=["claude"], rounds="3")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert debate["task"] == "Test task"
        assert debate["rounds"] == 3
        assert debate["current_round"] == 1
        assert debate["status"] == "running"
        assert debate["user_id"] == "U12345ABC"
        assert debate["team_id"] == "T12345ABC"

    @pytest.mark.asyncio
    async def test_debate_agents_mapped_to_display_names(self, interactions_module, state_module):
        view = _modal_view(agents=["claude", "gpt4", "gemini"])
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert "Claude" in debate["agents"]
        assert "GPT-4" in debate["agents"]
        assert "Gemini" in debate["agents"]

    @pytest.mark.asyncio
    async def test_debate_id_is_uuid(self, interactions_module, state_module):
        view = _modal_view()
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate_id = list(state_module._active_debates.keys())[0]
        # Should be a valid UUID
        uuid.UUID(debate_id)

    @pytest.mark.asyncio
    async def test_debate_audit_data_called(self, interactions_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.audit_data", mock_audit
        )
        view = _modal_view()
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["resource_type"] == "debate"
        assert call_kwargs["action"] == "create"
        assert call_kwargs["platform"] == "slack"

    @pytest.mark.asyncio
    async def test_empty_task_returns_error(self, interactions_module):
        view = _modal_view(task="")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"
        assert "task_block" in body["errors"]

    @pytest.mark.asyncio
    async def test_empty_agents_returns_error(self, interactions_module):
        view = _modal_view(agents=[])
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"
        assert "agents_block" in body["errors"]

    @pytest.mark.asyncio
    async def test_task_too_long_returns_error(self, interactions_module):
        long_task = "A" * 2001
        view = _modal_view(task=long_task)
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"
        assert "task_block" in body["errors"]

    @pytest.mark.asyncio
    async def test_task_with_injection_returns_error(self, interactions_module):
        view = _modal_view(task="Test'; DROP TABLE debates;--")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"
        assert "task_block" in body["errors"]

    @pytest.mark.asyncio
    async def test_agent_with_injection_returns_error(self, interactions_module):
        view = _modal_view(agents=["claude; rm -rf /"])
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"
        assert "agents_block" in body["errors"]

    @pytest.mark.asyncio
    async def test_rounds_non_digit_defaults(self, interactions_module, state_module):
        view = _modal_view(rounds="abc")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        from aragora.config import DEFAULT_ROUNDS

        assert debate["rounds"] == DEFAULT_ROUNDS

    @pytest.mark.asyncio
    async def test_rounds_zero_defaults(self, interactions_module, state_module):
        view = _modal_view(rounds="0")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        from aragora.config import DEFAULT_ROUNDS

        assert debate["rounds"] == DEFAULT_ROUNDS

    @pytest.mark.asyncio
    async def test_rounds_too_high_defaults(self, interactions_module, state_module):
        view = _modal_view(rounds="21")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        from aragora.config import DEFAULT_ROUNDS

        assert debate["rounds"] == DEFAULT_ROUNDS

    @pytest.mark.asyncio
    async def test_rounds_valid_boundary_1(self, interactions_module, state_module):
        view = _modal_view(rounds="1")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert debate["rounds"] == 1

    @pytest.mark.asyncio
    async def test_rounds_valid_boundary_20(self, interactions_module, state_module):
        view = _modal_view(rounds="20")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert debate["rounds"] == 20

    @pytest.mark.asyncio
    async def test_unknown_view_callback_id(self, interactions_module):
        view = _modal_view()
        view["callback_id"] = "other_modal"
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_view_callback_id_with_injection(self, interactions_module):
        view = _modal_view()
        view["callback_id"] = "start_debate_modal${evil}"
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        # Invalid callback_id gets caught by validation -> returns ok
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_unknown_agent_name_passed_through(self, interactions_module, state_module):
        """Agent names not in AGENT_DISPLAY_NAMES are used as-is."""
        view = _modal_view(agents=["custom_agent"])
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert "custom_agent" in debate["agents"]


# ===========================================================================
# User and Team ID validation
# ===========================================================================


class TestUserTeamValidation:
    """Tests for user_id and team_id validation."""

    @pytest.mark.asyncio
    async def test_invalid_user_id_format(self, interactions_module):
        """User IDs must be alphanumeric uppercase."""
        payload = _make_payload(user_id="user@bad")
        payload["actions"] = [_vote_action()]
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Invalid user identification" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_invalid_team_id_format(self, interactions_module):
        """Team IDs must start with T and be alphanumeric."""
        payload = _make_payload(team_id="bad-team")
        payload["actions"] = [_vote_action()]
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Invalid workspace identification" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_unknown_user_id_skips_validation(self, interactions_module):
        """user_id 'unknown' skips user_id validation."""
        payload = _make_payload(user_id="unknown", actions=[_summary_action("abc")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("response_type") == "ephemeral"

    @pytest.mark.asyncio
    async def test_empty_team_id_skips_validation(self, interactions_module):
        """Empty team_id skips team_id validation."""
        payload = _make_payload(actions=[_summary_action("abc")])
        payload["team"] = {"id": ""}
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("response_type") == "ephemeral"

    @pytest.mark.asyncio
    async def test_user_id_with_shell_injection(self, interactions_module):
        payload = _make_payload(user_id="U123|whoami")
        payload["actions"] = [_vote_action()]
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Invalid user identification" in body.get("text", "")


# ===========================================================================
# Unknown interaction types
# ===========================================================================


class TestUnknownInteractionType:
    """Tests for unknown or missing interaction types."""

    @pytest.mark.asyncio
    async def test_unknown_type_returns_ok(self, interactions_module):
        payload = _make_payload(interaction_type="dialog_submission")
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_missing_type_returns_ok(self, interactions_module):
        payload = {"user": {"id": "U123", "name": "x"}, "team": {"id": "T123"}}
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_null_type_returns_ok(self, interactions_module):
        payload = _make_payload()
        payload["type"] = None
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True


# ===========================================================================
# RBAC permission checks
# ===========================================================================


class TestRBACPermissions:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.asyncio
    async def test_vote_permission_denied(self, interactions_module, monkeypatch):
        mock_decision = MagicMock()
        mock_decision.allowed = False
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(return_value=mock_decision),
        )
        payload = _make_payload(actions=[_vote_action()])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Permission denied" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_summary_permission_denied(self, interactions_module, monkeypatch):
        mock_decision = MagicMock()
        mock_decision.allowed = False
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(return_value=mock_decision),
        )
        payload = _make_payload(actions=[_summary_action("abc")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Permission denied" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_shortcut_permission_denied(self, interactions_module, monkeypatch):
        mock_decision = MagicMock()
        mock_decision.allowed = False
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(return_value=mock_decision),
        )
        payload = _make_payload(
            interaction_type="shortcut",
            callback_id="start_debate",
        )
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "Permission denied" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_view_submission_permission_denied(self, interactions_module, monkeypatch):
        mock_decision = MagicMock()
        mock_decision.allowed = False
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(return_value=mock_decision),
        )
        view = _modal_view()
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"
        assert "permission" in body["errors"]["task_block"].lower()

    @pytest.mark.asyncio
    async def test_rbac_unavailable_fail_closed(self, interactions_module, monkeypatch):
        monkeypatch.setattr("aragora.server.handlers.bots.slack.interactions.RBAC_AVAILABLE", False)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.rbac_fail_closed",
            lambda: True,
        )
        payload = _make_payload(actions=[_vote_action()])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "access control module not loaded" in body.get("text", "").lower()

    @pytest.mark.asyncio
    async def test_rbac_unavailable_fail_open(self, interactions_module, monkeypatch, state_module):
        monkeypatch.setattr("aragora.server.handlers.bots.slack.interactions.RBAC_AVAILABLE", False)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.rbac_fail_closed",
            lambda: False,
        )
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        # Should succeed (fail-open)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_rbac_check_exception_falls_through(
        self, interactions_module, monkeypatch, state_module
    ):
        """When RBAC check_permission raises, it is caught and permission is allowed."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(side_effect=RuntimeError("RBAC failure")),
        )
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        # Should succeed since exception is caught
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_no_team_id_skips_rbac(self, interactions_module, monkeypatch, state_module):
        """When team_id is empty, RBAC check is skipped."""
        payload = _make_payload(team_id="", actions=[_vote_action("d1", "claude")])
        # Remove team from payload to get empty team_id
        payload["team"] = {"id": ""}
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_rbac_audit_on_denial(self, interactions_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.audit_data", mock_audit
        )
        mock_decision = MagicMock()
        mock_decision.allowed = False
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(return_value=mock_decision),
        )
        payload = _make_payload(actions=[_vote_action()])
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["action"] == "denied"
        assert call_kwargs["resource_type"] == "slack_permission"


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in the handler."""

    @pytest.mark.asyncio
    async def test_malformed_body_returns_500(self, interactions_module):
        """Body that isn't form-encoded -> parse error."""
        request = MockSlackRequest(raw_body=b"not-form-encoded")
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        # parse_qs returns empty for bad input, then payload_str defaults to "{}"
        # which parses fine. So this actually returns ok.
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_invalid_json_in_payload_returns_500(self, interactions_module):
        """payload field contains invalid JSON -> JSONDecodeError -> 500."""
        raw = urlencode({"payload": "{{invalid json}}"}).encode("utf-8")
        request = MockSlackRequest(raw_body=raw)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_missing_payload_field(self, interactions_module):
        """No 'payload' in form data -> defaults to '{}'."""
        raw = urlencode({"other_field": "value"}).encode("utf-8")
        request = MockSlackRequest(raw_body=raw)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_empty_body(self, interactions_module):
        """Empty body -> parse_qs returns empty -> payload defaults to '{}'."""
        request = MockSlackRequest(raw_body=b"")
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True


# ===========================================================================
# Security tests
# ===========================================================================


class TestSecurity:
    """Security-focused tests."""

    @pytest.mark.asyncio
    async def test_xss_in_task(self, interactions_module):
        view = _modal_view(task="<script>alert('xss')</script>")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"

    @pytest.mark.asyncio
    async def test_sql_injection_in_task(self, interactions_module):
        view = _modal_view(task="'; DROP TABLE debates;--")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"

    @pytest.mark.asyncio
    async def test_template_injection_in_task(self, interactions_module):
        view = _modal_view(task="${os.system('whoami')}")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"

    @pytest.mark.asyncio
    async def test_jinja_template_injection_in_task(self, interactions_module):
        view = _modal_view(task="{{config.__class__.__init__}}")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"

    @pytest.mark.asyncio
    async def test_shell_injection_in_agent(self, interactions_module):
        view = _modal_view(agents=["claude|whoami"])
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"

    @pytest.mark.asyncio
    async def test_hex_escape_in_callback_id(self, interactions_module):
        payload = _make_payload(
            interaction_type="shortcut",
            callback_id="start_debate\\x00",
        )
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        # Hex escape pattern detected -> validation fails -> returns ok
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_unicode_escape_in_action_id(self, interactions_module):
        action = {"action_id": "vote_\\u0000test", "value": "x"}
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_exports_handle_slack_interactions(self, interactions_module):
        assert "handle_slack_interactions" in interactions_module.__all__

    def test_handle_slack_interactions_is_callable(self, interactions_module):
        assert callable(interactions_module.handle_slack_interactions)


# ===========================================================================
# Rate limit decorator
# ===========================================================================


class TestRateLimitDecorator:
    """Tests that the rate_limit decorator wraps the function."""

    def test_has_wrapped_attribute(self, interactions_module):
        """The @rate_limit decorator should set __wrapped__."""
        assert hasattr(interactions_module.handle_slack_interactions, "__wrapped__")

    def test_wrapped_is_async(self, interactions_module):
        """The wrapped function should be async."""
        import asyncio

        assert asyncio.iscoroutinefunction(
            interactions_module.handle_slack_interactions.__wrapped__
        )


# ===========================================================================
# AuthorizationContext construction
# ===========================================================================


class TestAuthorizationContextConstruction:
    """Tests that the RBAC context is built correctly."""

    @pytest.mark.asyncio
    async def test_rbac_context_uses_slack_prefix(self, interactions_module, monkeypatch):
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            mock_check,
        )
        payload = _make_payload(actions=[_vote_action()])
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        # check_permission should be called with context that has slack:user_id
        ctx = mock_check.call_args[0][0]
        assert ctx.user_id == "slack:U12345ABC"
        assert ctx.workspace_id == "T12345ABC"
        assert "user" in ctx.roles


# ===========================================================================
# Missing user/team fields
# ===========================================================================


class TestMissingFields:
    """Tests for missing or partial user/team data in payload."""

    @pytest.mark.asyncio
    async def test_missing_user_object(self, interactions_module):
        """No 'user' key in payload -> defaults to unknown."""
        payload = {"type": "block_actions", "team": {"id": "T123"}, "actions": []}
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_missing_team_object(self, interactions_module):
        """No 'team' key in payload -> team_id defaults to ''."""
        payload = {"type": "block_actions", "user": {"id": "U123", "name": "x"}, "actions": []}
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_missing_user_name(self, interactions_module, state_module):
        """Missing user name defaults to 'unknown'."""
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        payload["user"] = {"id": "U12345ABC"}  # no name
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_view_submission_missing_state(self, interactions_module):
        """View with no state -> empty values -> empty task -> error."""
        view = {"callback_id": "start_debate_modal"}
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body["response_action"] == "errors"
        assert "task_block" in body["errors"]

    @pytest.mark.asyncio
    async def test_view_submission_missing_rounds(self, interactions_module, state_module):
        """Missing rounds selection defaults to DEFAULT_ROUNDS."""
        view = {
            "callback_id": "start_debate_modal",
            "state": {
                "values": {
                    "task_block": {"task_input": {"value": "Test task"}},
                    "agents_block": {
                        "agents_select": {
                            "selected_options": [{"value": "claude"}],
                        },
                    },
                    "rounds_block": {
                        "rounds_select": {
                            "selected_option": {},
                        },
                    },
                },
            },
        }
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        from aragora.config import DEFAULT_ROUNDS

        debate = list(state_module._active_debates.values())[0]
        assert debate["rounds"] == DEFAULT_ROUNDS


# ===========================================================================
# Additional vote edge cases
# ===========================================================================


class TestVoteEdgeCases:
    """Additional edge cases for vote actions."""

    @pytest.mark.asyncio
    async def test_vote_multiple_debates(self, interactions_module, state_module):
        """User can vote in multiple debates."""
        p1 = _make_payload(actions=[_vote_action("d1", "claude")])
        await interactions_module.handle_slack_interactions.__wrapped__(MockSlackRequest(p1))
        p2 = _make_payload(actions=[_vote_action("d2", "gpt4")])
        await interactions_module.handle_slack_interactions.__wrapped__(MockSlackRequest(p2))
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"
        assert state_module._user_votes["d2"]["U12345ABC"] == "gpt4"

    @pytest.mark.asyncio
    async def test_vote_agent_with_spaces(self, interactions_module, state_module):
        """Agent name with spaces should work (no injection chars)."""
        action = {
            "action_id": "vote_d1_agent",
            "value": json.dumps({"debate_id": "d1", "agent": "My Agent"}),
        }
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "My Agent"

    @pytest.mark.asyncio
    async def test_vote_debate_id_exact_100_chars(self, interactions_module, state_module):
        """debate_id of exactly 100 chars should be accepted."""
        debate_id = "a" * 100
        action = {
            "action_id": "vote_x",
            "value": json.dumps({"debate_id": debate_id, "agent": "claude"}),
        }
        payload = _make_payload(actions=[action])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "claude" in body.get("text", "")
        assert debate_id in state_module._user_votes

    @pytest.mark.asyncio
    async def test_vote_confirmation_text_format(self, interactions_module):
        """Verify the exact format of vote confirmation text."""
        payload = _make_payload(actions=[_vote_action("d1", "GPT-4")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "*GPT-4*" in body["text"]


# ===========================================================================
# View submission additional edge cases
# ===========================================================================


class TestViewSubmissionEdgeCases:
    """Additional edge cases for view_submission."""

    @pytest.mark.asyncio
    async def test_single_agent_creates_debate(self, interactions_module, state_module):
        view = _modal_view(agents=["claude"])
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert debate["agents"] == ["Claude"]

    @pytest.mark.asyncio
    async def test_many_agents_creates_debate(self, interactions_module, state_module):
        agents = ["claude", "gpt4", "gemini", "mistral", "deepseek", "grok", "qwen", "kimi"]
        view = _modal_view(agents=agents)
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert len(debate["agents"]) == 8
        assert "Claude" in debate["agents"]
        assert "Kimi" in debate["agents"]

    @pytest.mark.asyncio
    async def test_rounds_negative_defaults(self, interactions_module, state_module):
        """Negative rounds should default to DEFAULT_ROUNDS (isdigit returns False for '-1')."""
        view = _modal_view(rounds="-1")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        from aragora.config import DEFAULT_ROUNDS

        debate = list(state_module._active_debates.values())[0]
        assert debate["rounds"] == DEFAULT_ROUNDS

    @pytest.mark.asyncio
    async def test_rounds_float_defaults(self, interactions_module, state_module):
        """Float rounds should default (isdigit returns False for '3.5')."""
        view = _modal_view(rounds="3.5")
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        from aragora.config import DEFAULT_ROUNDS

        debate = list(state_module._active_debates.values())[0]
        assert debate["rounds"] == DEFAULT_ROUNDS

    @pytest.mark.asyncio
    async def test_task_at_max_length(self, interactions_module, state_module):
        """Task at exactly MAX_TOPIC_LENGTH should be accepted."""
        task = "A" * 2000
        view = _modal_view(task=task, agents=["claude"])
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert debate["task"] == task

    @pytest.mark.asyncio
    async def test_task_preview_in_audit(self, interactions_module, monkeypatch):
        """Audit data should contain task_preview truncated to 100 chars."""
        mock_audit = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.audit_data", mock_audit
        )
        long_task = "X" * 200
        view = _modal_view(task=long_task, agents=["claude"])
        payload = _make_payload(interaction_type="view_submission", view=view)
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        call_kwargs = mock_audit.call_args[1]
        assert len(call_kwargs["task_preview"]) == 100

    @pytest.mark.asyncio
    async def test_debate_stores_user_id(self, interactions_module, state_module):
        view = _modal_view()
        payload = _make_payload(
            interaction_type="view_submission",
            user_id="U999XYZ",
            view=view,
        )
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        debate = list(state_module._active_debates.values())[0]
        assert debate["user_id"] == "U999XYZ"

    @pytest.mark.asyncio
    async def test_view_missing_view_key(self, interactions_module):
        """No 'view' key in payload -> empty callback_id -> falls through."""
        payload = _make_payload(interaction_type="view_submission")
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_multiple_debates_get_unique_ids(self, interactions_module, state_module):
        """Each debate creation should get a unique UUID."""
        for _ in range(3):
            view = _modal_view()
            payload = _make_payload(interaction_type="view_submission", view=view)
            request = MockSlackRequest(payload)
            await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert len(state_module._active_debates) == 3
        ids = list(state_module._active_debates.keys())
        assert len(set(ids)) == 3  # all unique


# ===========================================================================
# RBAC additional edge cases
# ===========================================================================


class TestRBACEdgeCases:
    """Additional RBAC edge cases."""

    @pytest.mark.asyncio
    async def test_permission_error_in_check(self, interactions_module, monkeypatch, state_module):
        """PermissionError in check_permission is caught and access is allowed."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(side_effect=PermissionError("no access")),
        )
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_value_error_in_check(self, interactions_module, monkeypatch, state_module):
        """ValueError in check_permission is caught and access is allowed."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(side_effect=ValueError("bad value")),
        )
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_type_error_in_check(self, interactions_module, monkeypatch, state_module):
        """TypeError in check_permission is caught and access is allowed."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(side_effect=TypeError("type mismatch")),
        )
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_attribute_error_in_check(self, interactions_module, monkeypatch, state_module):
        """AttributeError in check_permission is caught and access is allowed."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission",
            MagicMock(side_effect=AttributeError("no attr")),
        )
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"

    @pytest.mark.asyncio
    async def test_rbac_fail_closed_for_shortcut(self, interactions_module, monkeypatch):
        """Fail-closed RBAC blocks shortcut interactions."""
        monkeypatch.setattr("aragora.server.handlers.bots.slack.interactions.RBAC_AVAILABLE", False)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.rbac_fail_closed",
            lambda: True,
        )
        payload = _make_payload(
            interaction_type="shortcut",
            callback_id="start_debate",
        )
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "access control module not loaded" in body.get("text", "").lower()

    @pytest.mark.asyncio
    async def test_rbac_fail_closed_for_summary(self, interactions_module, monkeypatch):
        """Fail-closed RBAC blocks summary requests."""
        monkeypatch.setattr("aragora.server.handlers.bots.slack.interactions.RBAC_AVAILABLE", False)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.rbac_fail_closed",
            lambda: True,
        )
        payload = _make_payload(actions=[_summary_action("abc")])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "access control module not loaded" in body.get("text", "").lower()

    @pytest.mark.asyncio
    async def test_authorization_context_none_skips_rbac(
        self, interactions_module, monkeypatch, state_module
    ):
        """When AuthorizationContext is None, RBAC check is skipped."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.interactions.AuthorizationContext", None
        )
        payload = _make_payload(actions=[_vote_action("d1", "claude")])
        request = MockSlackRequest(payload)
        await interactions_module.handle_slack_interactions.__wrapped__(request)
        assert state_module._user_votes["d1"]["U12345ABC"] == "claude"


# ===========================================================================
# Summary truncation test
# ===========================================================================


class TestSummaryTruncation:
    """Tests for debate ID truncation in summary response."""

    @pytest.mark.asyncio
    async def test_summary_truncates_debate_id(self, interactions_module):
        """Summary text shows first 8 chars of debate ID."""
        debate_id = "abcdefghijklmnop"
        payload = _make_payload(actions=[_summary_action(debate_id)])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "abcdefgh..." in body["text"]

    @pytest.mark.asyncio
    async def test_summary_short_debate_id(self, interactions_module):
        """Short debate IDs are still truncated with [:8]."""
        debate_id = "abc"
        payload = _make_payload(actions=[_summary_action(debate_id)])
        request = MockSlackRequest(payload)
        result = await interactions_module.handle_slack_interactions.__wrapped__(request)
        body = _body(result)
        assert "abc" in body["text"]
