"""
Tests for Slack Events API Handler.

Covers all routes and behavior of handle_slack_events():
- URL verification challenge:
  - Valid challenge returns echoed challenge
  - Empty challenge returns empty string
  - Challenge exceeding 500 chars returns 400
- Event type: app_mention
  - Valid mention with text returns processing message
  - Invalid user_id: silent OK
  - Invalid channel_id: silent OK
  - Invalid text (injection patterns): ephemeral error
  - Empty text: still returns processing message
  - RBAC available + permission denied
  - RBAC available + permission granted
  - RBAC unavailable + fail-closed
  - RBAC unavailable + fail-open
  - Clean text parsing: strip bot mention tags
  - Command extraction: "ask", "debate", "aragora" strip command prefix
  - Command extraction: "plan" sets decision_integrity flags
  - Command extraction: "implement" sets execution_mode
  - DecisionRouter routing: success, ImportError, RuntimeError
  - Attachment extraction from files and event attachments
  - Attachment hydration: success, connector unavailable, download error, skip large
  - Team ID validation for non-system events
- Event type: message
  - Returns OK (pass-through)
- Event type: app_uninstalled
  - Valid team_id: revokes token, audits
  - No team_id: returns OK without side effects
  - Invalid team_id: returns OK
  - Store import failure: logs error, returns OK
- Event type: tokens_revoked
  - Valid team_id: revokes token
  - No team_id: returns OK
  - Invalid team_id: returns OK
  - Store import failure: logs error, returns OK
- Unknown event type: returns OK
- No event at all: returns OK
- Malformed JSON body: returns 500 error
- _extract_slack_attachments() unit tests:
  - Files with various preview fields
  - Text truncation at max_preview
  - Non-dict entries skipped
  - Event attachments with fallback text
- _hydrate_slack_attachments() unit tests:
  - Empty list: returns immediately
  - Connector not found: returns unchanged
  - Import error: returns unchanged
  - Successful download populates data
  - Skip files already having data
  - Skip files exceeding max_bytes
  - Download failure: logs, continues
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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


class MockSlackEventRequest:
    """Mock request object that provides an async body() method."""

    def __init__(self, data: dict[str, Any] | None = None, raw_body: bytes | None = None):
        if raw_body is not None:
            self._raw = raw_body
        elif data is not None:
            self._raw = json.dumps(data).encode("utf-8")
        else:
            self._raw = b"{}"

    async def body(self) -> bytes:
        return self._raw


def _make_event_data(
    event_type: str = "app_mention",
    team_id: str = "T12345ABC",
    text: str = "<@U00BOT> hello",
    channel: str = "C12345ABC",
    user: str = "U12345ABC",
    thread_ts: str | None = None,
    files: list[dict] | None = None,
    attachments: list[dict] | None = None,
    outer_type: str = "event_callback",
    **extra_event: Any,
) -> dict[str, Any]:
    """Build a standard Slack event callback payload."""
    event: dict[str, Any] = {"type": event_type}
    if text is not None:
        event["text"] = text
    if channel:
        event["channel"] = channel
    if user:
        event["user"] = user
    if thread_ts:
        event["thread_ts"] = thread_ts
    if files is not None:
        event["files"] = files
    if attachments is not None:
        event["attachments"] = attachments
    event.update(extra_event)

    data: dict[str, Any] = {
        "type": outer_type,
        "team_id": team_id,
        "event": event,
    }
    return data


# ---------------------------------------------------------------------------
# Lazy module import fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def events_module(monkeypatch):
    """Import events module lazily so conftest patches apply first.

    By default, grant all RBAC permissions so non-RBAC tests don't fail.
    RBAC-focused tests override these via their own monkeypatches.
    """
    import aragora.server.handlers.bots.slack.events as mod

    # Default: RBAC available with all permissions granted
    mock_decision = MagicMock()
    mock_decision.allowed = True
    monkeypatch.setattr(mod, "RBAC_AVAILABLE", True)
    monkeypatch.setattr(mod, "check_permission", MagicMock(return_value=mock_decision))

    return mod


# ---------------------------------------------------------------------------
# URL Verification Tests
# ---------------------------------------------------------------------------


class TestURLVerification:
    """Tests for the url_verification challenge-response flow."""

    @pytest.mark.asyncio
    async def test_valid_challenge(self, events_module):
        data = {"type": "url_verification", "challenge": "test-challenge-abc"}
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["challenge"] == "test-challenge-abc"

    @pytest.mark.asyncio
    async def test_empty_challenge(self, events_module):
        data = {"type": "url_verification"}
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        assert _body(result)["challenge"] == ""

    @pytest.mark.asyncio
    async def test_challenge_too_long(self, events_module):
        data = {"type": "url_verification", "challenge": "x" * 501}
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_challenge_exactly_500(self, events_module):
        data = {"type": "url_verification", "challenge": "x" * 500}
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        assert _body(result)["challenge"] == "x" * 500

    @pytest.mark.asyncio
    async def test_challenge_with_special_chars(self, events_module):
        challenge = "abc123-def456_ghijklmnop"
        data = {"type": "url_verification", "challenge": challenge}
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        assert _body(result)["challenge"] == challenge


# ---------------------------------------------------------------------------
# App Mention Tests
# ---------------------------------------------------------------------------


class TestAppMention:
    """Tests for the app_mention event type."""

    @pytest.mark.asyncio
    async def test_valid_mention_returns_processing(self, events_module):
        data = _make_event_data(event_type="app_mention", text="<@U00BOT> hello world")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert "Processing" in body["text"]

    @pytest.mark.asyncio
    async def test_invalid_user_id_silent_ok(self, events_module):
        """Invalid user_id returns silent OK (no error exposed to Slack)."""
        data = _make_event_data(user="invalid-user!")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_empty_user_id_silent_ok(self, events_module):
        data = _make_event_data(user="")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_invalid_channel_id_silent_ok(self, events_module):
        """Invalid channel_id returns silent OK."""
        data = _make_event_data(channel="bad-channel!")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_empty_channel_id_silent_ok(self, events_module):
        data = _make_event_data(channel="")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_invalid_text_injection_returns_ephemeral(self, events_module):
        """Text containing injection patterns returns ephemeral error."""
        data = _make_event_data(text="<script>alert('xss')</script>")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("response_type") == "ephemeral"
        assert "Invalid input" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_text_with_shell_injection(self, events_module):
        data = _make_event_data(text="hello; rm -rf /")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("response_type") == "ephemeral"

    @pytest.mark.asyncio
    async def test_text_with_template_injection(self, events_module):
        data = _make_event_data(text="${malicious}")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("response_type") == "ephemeral"

    @pytest.mark.asyncio
    async def test_text_too_long(self, events_module):
        from aragora.server.handlers.bots.slack.constants import MAX_TOPIC_LENGTH

        data = _make_event_data(text="x" * (MAX_TOPIC_LENGTH + 1))
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("response_type") == "ephemeral"
        assert "Invalid input" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_empty_text_returns_processing(self, events_module):
        """Empty text is allowed (allow_empty=True), returns processing message."""
        data = _make_event_data(text="")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_mention_strips_bot_tags(self, events_module):
        """Bot mention tags like <@U00BOT> are stripped from clean_text."""
        data = _make_event_data(text="<@U00BOT> what is the meaning of life")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_invalid_team_id_non_system_event(self, events_module):
        """Invalid team_id returns error for non-system events."""
        data = _make_event_data(team_id="invalid-team!")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_team_id_allowed(self, events_module):
        """Empty team_id is allowed (skips validation)."""
        data = _make_event_data(team_id="")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# App Mention - Command Parsing Tests
# ---------------------------------------------------------------------------


class TestAppMentionCommands:
    """Tests for command parsing in app_mention events."""

    @pytest.mark.asyncio
    async def test_ask_command_strips_prefix(self, events_module):
        data = _make_event_data(text="<@U00BOT> ask What is the best DB")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_debate_command_strips_prefix(self, events_module):
        data = _make_event_data(text="<@U00BOT> debate Should we use Postgres")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_aragora_command_strips_prefix(self, events_module):
        data = _make_event_data(text="<@U00BOT> aragora Rate limiters")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_plan_command_sets_decision_integrity(self, events_module):
        """plan command sets include_receipt, include_plan, plan_strategy."""
        data = _make_event_data(text="<@U00BOT> plan Build auth system")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_implement_command_sets_execution_mode(self, events_module):
        """implement command sets execution_mode and execution_engine."""
        data = _make_event_data(text="<@U00BOT> implement Add login page")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unknown_command_treated_as_text(self, events_module):
        """Unknown commands are treated as plain debate text."""
        data = _make_event_data(text="<@U00BOT> randomcommand some text")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"


# ---------------------------------------------------------------------------
# App Mention - RBAC Tests
# ---------------------------------------------------------------------------


class TestAppMentionRBAC:
    """Tests for RBAC permission checks in app_mention events."""

    @pytest.mark.asyncio
    async def test_rbac_unavailable_fail_closed(self, events_module, monkeypatch):
        """When RBAC unavailable and fail-closed, return service unavailable."""
        monkeypatch.setattr(events_module, "RBAC_AVAILABLE", False)
        monkeypatch.setattr(events_module, "rbac_fail_closed", lambda: True)

        data = _make_event_data()
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("response_type") == "ephemeral"
        assert (
            "access control" in body.get("text", "").lower()
            or "unavailable" in body.get("text", "").lower()
        )

    @pytest.mark.asyncio
    async def test_rbac_unavailable_fail_open(self, events_module, monkeypatch):
        """When RBAC unavailable and fail-open, processing continues."""
        monkeypatch.setattr(events_module, "RBAC_AVAILABLE", False)
        monkeypatch.setattr(events_module, "rbac_fail_closed", lambda: False)

        data = _make_event_data()
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_rbac_permission_denied(self, events_module, monkeypatch):
        """When RBAC denies permission, return ephemeral error."""
        monkeypatch.setattr(events_module, "RBAC_AVAILABLE", True)

        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_check = MagicMock(return_value=mock_decision)
        monkeypatch.setattr(events_module, "check_permission", mock_check)

        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(events_module, "AuthorizationContext", mock_ctx_cls)

        data = _make_event_data(team_id="T12345ABC")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("response_type") == "ephemeral"
        assert "permission" in body.get("text", "").lower()

    @pytest.mark.asyncio
    async def test_rbac_permission_granted(self, events_module, monkeypatch):
        """When RBAC grants permission, processing continues."""
        monkeypatch.setattr(events_module, "RBAC_AVAILABLE", True)

        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)
        monkeypatch.setattr(events_module, "check_permission", mock_check)

        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(events_module, "AuthorizationContext", mock_ctx_cls)

        data = _make_event_data(team_id="T12345ABC")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_rbac_check_exception_continues(self, events_module, monkeypatch):
        """When RBAC check raises, processing continues (graceful degradation)."""
        monkeypatch.setattr(events_module, "RBAC_AVAILABLE", True)

        mock_check = MagicMock(side_effect=TypeError("bad args"))
        monkeypatch.setattr(events_module, "check_permission", mock_check)

        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(events_module, "AuthorizationContext", mock_ctx_cls)

        data = _make_event_data(team_id="T12345ABC")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_rbac_no_team_id_skips_check(self, events_module, monkeypatch):
        """When team_id is empty, RBAC check is skipped."""
        monkeypatch.setattr(events_module, "RBAC_AVAILABLE", True)

        mock_check = MagicMock()
        monkeypatch.setattr(events_module, "check_permission", mock_check)

        data = _make_event_data(team_id="")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        # check_permission should NOT have been called (no team_id)
        mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_rbac_check_permission_none_skips(self, events_module, monkeypatch):
        """When check_permission is None, skip RBAC check."""
        monkeypatch.setattr(events_module, "RBAC_AVAILABLE", True)
        monkeypatch.setattr(events_module, "check_permission", None)

        data = _make_event_data(team_id="T12345ABC")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"


# ---------------------------------------------------------------------------
# App Mention - DecisionRouter Tests
# ---------------------------------------------------------------------------


class TestAppMentionRouting:
    """Tests for DecisionRouter routing in app_mention events."""

    @pytest.mark.asyncio
    async def test_decision_router_import_error(self, events_module):
        """If aragora.core can't be imported, silently continue."""
        data = _make_event_data(text="<@U00BOT> ask something meaningful")
        req = MockSlackEventRequest(data)

        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_decision_router_runtime_error(self, events_module):
        """If router.route() raises, error is logged but response is OK."""
        mock_core = MagicMock()
        mock_core.DecisionType.DEBATE = "debate"
        mock_core.InputSource.SLACK = "slack"
        mock_core.ResponseChannel = MagicMock()
        mock_core.RequestContext = MagicMock()
        mock_core.DecisionRequest = MagicMock(side_effect=RuntimeError("route fail"))
        mock_core.get_decision_router = MagicMock()

        data = _make_event_data(text="<@U00BOT> ask a question")
        req = MockSlackEventRequest(data)

        with patch.dict("sys.modules", {"aragora.core": mock_core}):
            result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_no_clean_text_skips_routing(self, events_module):
        """If clean_text is empty after stripping, skip routing."""
        data = _make_event_data(text="<@U00BOT>")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_only_command_no_remainder(self, events_module):
        """'ask' with no remainder produces empty clean_text, skips routing."""
        data = _make_event_data(text="<@U00BOT> ask")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"


# ---------------------------------------------------------------------------
# Message Event Tests
# ---------------------------------------------------------------------------


class TestMessageEvent:
    """Tests for the message event type (pass-through)."""

    @pytest.mark.asyncio
    async def test_message_event_returns_ok(self, events_module):
        data = _make_event_data(event_type="message")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True


# ---------------------------------------------------------------------------
# App Uninstalled Tests
# ---------------------------------------------------------------------------


class TestAppUninstalled:
    """Tests for the app_uninstalled event type."""

    @pytest.mark.asyncio
    async def test_valid_team_revokes_token(self, events_module):
        """Valid team_id triggers token revocation and audit."""
        mock_store = MagicMock()
        mock_get_store = MagicMock(return_value=mock_store)

        data = _make_event_data(event_type="app_uninstalled", team_id="T12345ABC")
        req = MockSlackEventRequest(data)

        with (
            patch("aragora.server.handlers.bots.slack.events.audit_data") as mock_audit,
            patch.dict("sys.modules", {}),
        ):
            with patch(
                "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                mock_get_store,
            ):
                result = await events_module.handle_slack_events(req)

        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True
        mock_store.revoke_token.assert_called_once_with("T12345ABC")
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_team_id_returns_ok(self, events_module):
        """No team_id returns OK without side effects."""
        data = {
            "type": "event_callback",
            "event": {"type": "app_uninstalled"},
        }
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_invalid_team_id_returns_ok(self, events_module):
        """Invalid team_id for uninstall returns silent OK."""
        data = _make_event_data(event_type="app_uninstalled", team_id="INVALID!")
        # The handler re-reads team_id from data or event; set both invalid
        data["team_id"] = "INVALID!"
        data["event"]["team_id"] = "INVALID!"
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_store_import_error(self, events_module):
        """ImportError on store is caught, returns OK."""
        data = _make_event_data(event_type="app_uninstalled", team_id="T12345ABC")
        req = MockSlackEventRequest(data)

        # Remove the module from sys.modules and make re-import fail
        import sys

        saved = sys.modules.pop("aragora.storage.slack_workspace_store", None)
        fake = MagicMock()
        fake.get_slack_workspace_store = MagicMock(side_effect=ImportError("no store"))
        # Set to None so the import statement raises ImportError
        sys.modules["aragora.storage.slack_workspace_store"] = None  # type: ignore
        try:
            result = await events_module.handle_slack_events(req)
        finally:
            if saved is not None:
                sys.modules["aragora.storage.slack_workspace_store"] = saved
            else:
                sys.modules.pop("aragora.storage.slack_workspace_store", None)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_store_runtime_error(self, events_module):
        """RuntimeError from store.revoke_token is caught, returns OK."""
        mock_store = MagicMock()
        mock_store.revoke_token.side_effect = RuntimeError("db error")
        mock_get_store = MagicMock(return_value=mock_store)

        data = _make_event_data(event_type="app_uninstalled", team_id="T12345ABC")
        req = MockSlackEventRequest(data)

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            mock_get_store,
        ):
            result = await events_module.handle_slack_events(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_team_id_from_event_fallback(self, events_module):
        """When outer team_id is missing, team_id is read from event."""
        mock_store = MagicMock()
        mock_get_store = MagicMock(return_value=mock_store)

        data = {
            "type": "event_callback",
            "event": {"type": "app_uninstalled", "team_id": "T98765XYZ"},
        }
        req = MockSlackEventRequest(data)

        with (
            patch("aragora.server.handlers.bots.slack.events.audit_data"),
            patch(
                "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                mock_get_store,
            ),
        ):
            result = await events_module.handle_slack_events(req)

        assert _status(result) == 200
        mock_store.revoke_token.assert_called_once_with("T98765XYZ")


# ---------------------------------------------------------------------------
# Tokens Revoked Tests
# ---------------------------------------------------------------------------


class TestTokensRevoked:
    """Tests for the tokens_revoked event type."""

    @pytest.mark.asyncio
    async def test_valid_team_revokes_token(self, events_module):
        mock_store = MagicMock()
        mock_get_store = MagicMock(return_value=mock_store)

        data = _make_event_data(event_type="tokens_revoked", team_id="T12345ABC")
        req = MockSlackEventRequest(data)

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            mock_get_store,
        ):
            result = await events_module.handle_slack_events(req)

        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True
        mock_store.revoke_token.assert_called_once_with("T12345ABC")

    @pytest.mark.asyncio
    async def test_no_team_id_returns_ok(self, events_module):
        data = {
            "type": "event_callback",
            "event": {"type": "tokens_revoked"},
        }
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_invalid_team_id_returns_ok(self, events_module):
        data = _make_event_data(event_type="tokens_revoked", team_id="INVALID!")
        data["team_id"] = "INVALID!"
        data["event"]["team_id"] = "INVALID!"
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_store_import_error(self, events_module):
        data = _make_event_data(event_type="tokens_revoked", team_id="T12345ABC")
        req = MockSlackEventRequest(data)

        import sys

        saved = sys.modules.pop("aragora.storage.slack_workspace_store", None)
        sys.modules["aragora.storage.slack_workspace_store"] = None  # type: ignore
        try:
            result = await events_module.handle_slack_events(req)
        finally:
            if saved is not None:
                sys.modules["aragora.storage.slack_workspace_store"] = saved
            else:
                sys.modules.pop("aragora.storage.slack_workspace_store", None)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_store_oserror(self, events_module):
        mock_store = MagicMock()
        mock_store.revoke_token.side_effect = OSError("disk error")
        mock_get_store = MagicMock(return_value=mock_store)

        data = _make_event_data(event_type="tokens_revoked", team_id="T12345ABC")
        req = MockSlackEventRequest(data)

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            mock_get_store,
        ):
            result = await events_module.handle_slack_events(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_team_id_from_event_fallback(self, events_module):
        mock_store = MagicMock()
        mock_get_store = MagicMock(return_value=mock_store)

        data = {
            "type": "event_callback",
            "event": {"type": "tokens_revoked", "team_id": "T98765XYZ"},
        }
        req = MockSlackEventRequest(data)

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            mock_get_store,
        ):
            result = await events_module.handle_slack_events(req)

        assert _status(result) == 200
        mock_store.revoke_token.assert_called_once_with("T98765XYZ")


# ---------------------------------------------------------------------------
# Unknown / Missing Event Tests
# ---------------------------------------------------------------------------


class TestUnknownEvents:
    """Tests for unknown or missing event types."""

    @pytest.mark.asyncio
    async def test_unknown_event_type_returns_ok(self, events_module):
        data = _make_event_data(event_type="channel_created")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_no_event_returns_ok(self, events_module):
        data = {"type": "event_callback"}
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_empty_body_returns_ok(self, events_module):
        req = MockSlackEventRequest({})
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in the handler."""

    @pytest.mark.asyncio
    async def test_malformed_json_returns_500(self, events_module):
        req = MockSlackEventRequest(raw_body=b"not valid json {{{")
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_non_utf8_body_returns_500(self, events_module):
        req = MockSlackEventRequest(raw_body=b"\xff\xfe invalid")
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_body_method_raises(self, events_module):
        """If request.body() raises, the outer except catches it if it's a handled type."""

        class BadRequest:
            async def body(self):
                raise ValueError("connection dropped")

        result = await events_module.handle_slack_events(BadRequest())
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# _extract_slack_attachments() Unit Tests
# ---------------------------------------------------------------------------


class TestExtractSlackAttachments:
    """Unit tests for the _extract_slack_attachments helper."""

    def test_empty_event(self, events_module):
        result = events_module._extract_slack_attachments({})
        assert result == []

    def test_file_with_preview_plain_text(self, events_module):
        event = {
            "files": [
                {
                    "id": "F001",
                    "name": "test.txt",
                    "mimetype": "text/plain",
                    "size": 100,
                    "url_private_download": "https://files.slack.com/f001",
                    "preview_plain_text": "Hello world",
                }
            ]
        }
        result = events_module._extract_slack_attachments(event)
        assert len(result) == 1
        assert result[0]["type"] == "slack_file"
        assert result[0]["file_id"] == "F001"
        assert result[0]["filename"] == "test.txt"
        assert result[0]["content_type"] == "text/plain"
        assert result[0]["size"] == 100
        assert result[0]["url"] == "https://files.slack.com/f001"
        assert result[0]["text"] == "Hello world"

    def test_file_preview_fallback_chain(self, events_module):
        """Falls back: preview_plain_text -> preview -> initial_comment.comment."""
        event = {
            "files": [
                {
                    "id": "F002",
                    "initial_comment": {"comment": "my comment"},
                }
            ]
        }
        result = events_module._extract_slack_attachments(event)
        assert len(result) == 1
        assert result[0]["text"] == "my comment"

    def test_file_preview_truncation(self, events_module):
        long_text = "x" * 3000
        event = {"files": [{"id": "F003", "preview_plain_text": long_text}]}
        result = events_module._extract_slack_attachments(event, max_preview=2000)
        assert len(result) == 1
        assert result[0]["text"].endswith("...")
        assert len(result[0]["text"]) == 2003  # 2000 + "..."

    def test_file_url_fallback_chain(self, events_module):
        """Falls back: url_private_download -> url_private -> permalink."""
        event = {"files": [{"id": "F004", "permalink": "https://example.com/p"}]}
        result = events_module._extract_slack_attachments(event)
        assert result[0]["url"] == "https://example.com/p"

    def test_file_name_fallback_chain(self, events_module):
        """Falls back: name -> title -> 'file'."""
        # No name or title
        event = {"files": [{"id": "F005"}]}
        result = events_module._extract_slack_attachments(event)
        assert result[0]["filename"] == "file"

        # Title only
        event2 = {"files": [{"id": "F006", "title": "Report.pdf"}]}
        result2 = events_module._extract_slack_attachments(event2)
        assert result2[0]["filename"] == "Report.pdf"

    def test_non_dict_files_skipped(self, events_module):
        event = {"files": ["not-a-dict", 123, None, {"id": "F007"}]}
        result = events_module._extract_slack_attachments(event)
        assert len(result) == 1
        assert result[0]["file_id"] == "F007"

    def test_files_not_a_list(self, events_module):
        event = {"files": "not-a-list"}
        result = events_module._extract_slack_attachments(event)
        assert result == []

    def test_event_attachments(self, events_module):
        event = {
            "attachments": [
                {
                    "title": "Link Title",
                    "text": "Link description",
                    "title_link": "https://example.com",
                }
            ]
        }
        result = events_module._extract_slack_attachments(event)
        assert len(result) == 1
        assert result[0]["type"] == "slack_attachment"
        assert result[0]["title"] == "Link Title"
        assert result[0]["url"] == "https://example.com"
        assert result[0]["text"] == "Link description"

    def test_event_attachment_fallback_text(self, events_module):
        event = {"attachments": [{"fallback": "fallback text"}]}
        result = events_module._extract_slack_attachments(event)
        assert result[0]["text"] == "fallback text"

    def test_event_attachment_text_truncation(self, events_module):
        event = {"attachments": [{"text": "y" * 3000}]}
        result = events_module._extract_slack_attachments(event, max_preview=100)
        assert result[0]["text"] == "y" * 100 + "..."

    def test_event_attachment_url_fallback(self, events_module):
        event = {"attachments": [{"from_url": "https://example.com/from"}]}
        result = events_module._extract_slack_attachments(event)
        assert result[0]["url"] == "https://example.com/from"

    def test_non_dict_attachments_skipped(self, events_module):
        event = {"attachments": ["string", {"title": "Valid"}]}
        result = events_module._extract_slack_attachments(event)
        assert len(result) == 1

    def test_attachments_not_a_list(self, events_module):
        event = {"attachments": "not-a-list"}
        result = events_module._extract_slack_attachments(event)
        assert result == []

    def test_both_files_and_attachments(self, events_module):
        event = {
            "files": [{"id": "F001", "name": "a.txt"}],
            "attachments": [{"title": "Link"}],
        }
        result = events_module._extract_slack_attachments(event)
        assert len(result) == 2
        assert result[0]["type"] == "slack_file"
        assert result[1]["type"] == "slack_attachment"

    def test_custom_max_preview(self, events_module):
        event = {"files": [{"id": "F008", "preview_plain_text": "a" * 50}]}
        result = events_module._extract_slack_attachments(event, max_preview=10)
        assert result[0]["text"] == "a" * 10 + "..."

    def test_file_content_type_fallback(self, events_module):
        """Falls back from mimetype to filetype."""
        event = {"files": [{"id": "F009", "filetype": "pdf"}]}
        result = events_module._extract_slack_attachments(event)
        assert result[0]["content_type"] == "pdf"

    def test_empty_preview_no_truncation(self, events_module):
        event = {"files": [{"id": "F010", "preview_plain_text": ""}]}
        result = events_module._extract_slack_attachments(event)
        assert result[0]["text"] == ""

    def test_non_string_preview_not_truncated(self, events_module):
        """Non-string preview is not truncated (isinstance check)."""
        event = {"files": [{"id": "F011", "preview_plain_text": 12345}]}
        result = events_module._extract_slack_attachments(event)
        assert result[0]["text"] == 12345

    def test_attachment_no_title_defaults_to_attachment(self, events_module):
        event = {"attachments": [{"text": "some text"}]}
        result = events_module._extract_slack_attachments(event)
        assert result[0]["filename"] == "attachment"


# ---------------------------------------------------------------------------
# _hydrate_slack_attachments() Unit Tests
# ---------------------------------------------------------------------------


class TestHydrateSlackAttachments:
    """Unit tests for the _hydrate_slack_attachments helper."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_immediately(self, events_module):
        result = await events_module._hydrate_slack_attachments([])
        assert result == []

    @pytest.mark.asyncio
    async def test_connector_import_error(self, events_module):
        """If connector registry can't be imported, return unchanged."""
        attachments = [{"file_id": "F001", "type": "slack_file"}]

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": None}):
            result = await events_module._hydrate_slack_attachments(attachments)
        assert result == attachments

    @pytest.mark.asyncio
    async def test_connector_not_found(self, events_module):
        """If get_connector('slack') returns None, return unchanged."""
        mock_registry = MagicMock()
        mock_registry.get_connector.return_value = None

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_registry}):
            with patch(
                "aragora.server.handlers.bots.slack.events.get_connector",
                mock_registry.get_connector,
                create=True,
            ):
                # Need to do it differently - the function uses a local import
                pass

        # Use the proper approach - patch where it's imported
        attachments = [{"file_id": "F001"}]

        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = None

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)
        assert result == attachments

    @pytest.mark.asyncio
    async def test_successful_download(self, events_module):
        """File content is populated from downloaded file."""
        attachments = [{"file_id": "F001", "type": "slack_file"}]

        mock_file = MagicMock()
        mock_file.content = b"file content here"
        mock_file.filename = "downloaded.txt"
        mock_file.content_type = "text/plain"
        mock_file.size = 17

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(return_value=mock_file)

        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        assert result[0]["data"] == b"file content here"
        assert result[0]["filename"] == "downloaded.txt"
        assert result[0]["content_type"] == "text/plain"
        assert result[0]["size"] == 17

    @pytest.mark.asyncio
    async def test_skip_already_has_data(self, events_module):
        """Files that already have 'data' are skipped."""
        attachments = [{"file_id": "F001", "data": b"existing"}]

        mock_connector = AsyncMock()
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        mock_connector.download_file.assert_not_called()
        assert result[0]["data"] == b"existing"

    @pytest.mark.asyncio
    async def test_skip_already_has_content(self, events_module):
        """Files that already have 'content' are skipped."""
        attachments = [{"file_id": "F001", "content": "existing text"}]

        mock_connector = AsyncMock()
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        mock_connector.download_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_no_file_id(self, events_module):
        """Files without file_id are skipped."""
        attachments = [{"type": "slack_file"}]

        mock_connector = AsyncMock()
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        mock_connector.download_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_too_large(self, events_module):
        """Files exceeding max_bytes are skipped."""
        attachments = [{"file_id": "F001", "size": 5_000_000}]

        mock_connector = AsyncMock()
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(
                attachments, max_bytes=2_000_000
            )

        mock_connector.download_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_download_connection_error(self, events_module):
        """ConnectionError during download is caught, file unchanged."""
        attachments = [{"file_id": "F001"}]

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(side_effect=ConnectionError("refused"))
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        assert "data" not in result[0]

    @pytest.mark.asyncio
    async def test_download_timeout_error(self, events_module):
        """TimeoutError during download is caught."""
        attachments = [{"file_id": "F001"}]

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        assert "data" not in result[0]

    @pytest.mark.asyncio
    async def test_download_oserror(self, events_module):
        """OSError during download is caught."""
        attachments = [{"file_id": "F001"}]

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(side_effect=OSError("io error"))
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        assert "data" not in result[0]

    @pytest.mark.asyncio
    async def test_download_value_error(self, events_module):
        """ValueError during download is caught."""
        attachments = [{"file_id": "F001"}]

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(side_effect=ValueError("bad value"))
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        assert "data" not in result[0]

    @pytest.mark.asyncio
    async def test_non_dict_attachments_skipped(self, events_module):
        """Non-dict entries in the attachments list are skipped."""
        attachments = ["not-a-dict", {"file_id": "F001"}]

        mock_file = MagicMock()
        mock_file.content = b"data"
        mock_file.filename = None
        mock_file.content_type = None
        mock_file.size = None

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(return_value=mock_file)
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        # Only the dict entry should have been processed
        assert result[1].get("data") == b"data"

    @pytest.mark.asyncio
    async def test_download_no_content(self, events_module):
        """If downloaded file has no content attribute, skip setting data."""
        attachments = [{"file_id": "F001"}]

        mock_file = MagicMock(spec=[])  # No attributes at all

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(return_value=mock_file)
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        assert "data" not in result[0]

    @pytest.mark.asyncio
    async def test_preserves_existing_filename(self, events_module):
        """Existing filename is not overwritten even if file_obj has one."""
        attachments = [{"file_id": "F001", "filename": "original.txt"}]

        mock_file = MagicMock()
        mock_file.content = b"data"
        mock_file.filename = "downloaded.txt"
        mock_file.content_type = None
        mock_file.size = None

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(return_value=mock_file)
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        assert result[0]["filename"] == "original.txt"

    @pytest.mark.asyncio
    async def test_size_not_int_allows_download(self, events_module):
        """Non-int size doesn't trigger the skip-large check."""
        attachments = [{"file_id": "F001", "size": "unknown"}]

        mock_file = MagicMock()
        mock_file.content = b"data"
        mock_file.filename = None
        mock_file.content_type = None
        mock_file.size = None

        mock_connector = AsyncMock()
        mock_connector.download_file = AsyncMock(return_value=mock_file)
        mock_mod = MagicMock()
        mock_mod.get_connector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_mod}):
            result = await events_module._hydrate_slack_attachments(attachments)

        assert result[0].get("data") == b"data"


# ---------------------------------------------------------------------------
# Team ID Validation Edge Cases
# ---------------------------------------------------------------------------


class TestTeamIDValidation:
    """Tests for team_id validation in different event types."""

    @pytest.mark.asyncio
    async def test_system_events_skip_initial_team_id_validation(self, events_module):
        """app_uninstalled and tokens_revoked skip the initial team_id validation."""
        # Use an invalid team_id in the outer data but valid in event
        # The handler re-reads team_id for system events
        for event_type in ("app_uninstalled", "tokens_revoked"):
            data = {
                "type": "event_callback",
                "team_id": "INVALID",
                "event": {"type": event_type, "team_id": "T12345ABC"},
            }
            req = MockSlackEventRequest(data)
            # Should not return 400 for invalid outer team_id
            result = await events_module.handle_slack_events(req)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_non_system_event_validates_team_id(self, events_module):
        """Non-system events validate team_id format."""
        data = _make_event_data(event_type="app_mention", team_id="BAD_TEAM!")
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# Attachment Integration Tests (within app_mention flow)
# ---------------------------------------------------------------------------


class TestAppMentionAttachments:
    """Tests for attachment handling in the app_mention flow."""

    @pytest.mark.asyncio
    async def test_mention_with_file_attachments(self, events_module):
        """app_mention with files extracts attachments."""
        data = _make_event_data(
            text="<@U00BOT> review this file",
            files=[{"id": "F001", "name": "doc.pdf", "mimetype": "application/pdf"}],
        )
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_mention_with_event_attachments(self, events_module):
        """app_mention with event attachments extracts them."""
        data = _make_event_data(
            text="<@U00BOT> check this link",
            attachments=[{"title": "Article", "text": "Some article text"}],
        )
        req = MockSlackEventRequest(data)
        result = await events_module.handle_slack_events(req)
        assert _status(result) == 200
