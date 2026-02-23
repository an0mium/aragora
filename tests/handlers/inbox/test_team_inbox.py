"""Tests for team inbox handler (aragora/server/handlers/inbox/team_inbox.py).

Covers all 12 handler functions and the registration helper:
- handle_get_team_members          GET    /api/v1/inbox/shared/{id}/team
- handle_add_team_member           POST   /api/v1/inbox/shared/{id}/team
- handle_remove_team_member        DELETE /api/v1/inbox/shared/{id}/team/{user_id}
- handle_start_viewing             POST   /api/v1/inbox/shared/{id}/messages/{msg_id}/viewing
- handle_stop_viewing              DELETE /api/v1/inbox/shared/{id}/messages/{msg_id}/viewing
- handle_start_typing              POST   /api/v1/inbox/shared/{id}/messages/{msg_id}/typing
- handle_stop_typing               DELETE /api/v1/inbox/shared/{id}/messages/{msg_id}/typing
- handle_get_notes                 GET    /api/v1/inbox/shared/{id}/messages/{msg_id}/notes
- handle_add_note                  POST   /api/v1/inbox/shared/{id}/messages/{msg_id}/notes
- handle_get_mentions              GET    /api/v1/inbox/mentions
- handle_acknowledge_mention       POST   /api/v1/inbox/mentions/{id}/acknowledge
- handle_get_activity_feed         GET    /api/v1/inbox/shared/{id}/activity
- get_team_inbox_handlers          Registration helper
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.inbox.team_inbox import (
    get_team_inbox_handlers,
    handle_acknowledge_mention,
    handle_add_note,
    handle_add_team_member,
    handle_get_activity_feed,
    handle_get_mentions,
    handle_get_notes,
    handle_get_team_members,
    handle_remove_team_member,
    handle_start_typing,
    handle_start_viewing,
    handle_stop_typing,
    handle_stop_viewing,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------


class MockTeamMember:
    """Mock team member returned by emitter."""

    def __init__(
        self,
        user_id: str = "user-1",
        email: str = "user@example.com",
        name: str = "Test User",
        role: str = "member",
    ):
        self.user_id = user_id
        self.email = email
        self.name = name
        self.role = role

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
        }


class MockNote:
    """Mock note returned by emitter."""

    def __init__(
        self,
        note_id: str = "note-1",
        content: str = "A note",
        author_id: str = "user-1",
        author_name: str = "Test User",
    ):
        self.id = note_id
        self.content = content
        self.author_id = author_id
        self.author_name = author_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "author_id": self.author_id,
            "author_name": self.author_name,
        }


class MockMention:
    """Mock mention returned by emitter."""

    def __init__(
        self,
        mention_id: str = "mention-1",
        mentioned_user_id: str = "user-1",
        acknowledged: bool = False,
    ):
        self.mention_id = mention_id
        self.mentioned_user_id = mentioned_user_id
        self.acknowledged = acknowledged

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "mentioned_user_id": self.mentioned_user_id,
            "acknowledged": self.acknowledged,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_emitter():
    """Create a mock team inbox emitter with all methods wired."""
    emitter = AsyncMock()

    # Team member management
    emitter.get_team_members = AsyncMock(return_value=[])
    emitter.add_team_member = AsyncMock(return_value=MockTeamMember())
    emitter.remove_team_member = AsyncMock(return_value=True)

    # Presence
    emitter.emit_user_viewing = AsyncMock()
    emitter.emit_user_left = AsyncMock()
    emitter.emit_user_typing = AsyncMock()
    emitter.emit_user_stopped_typing = AsyncMock()
    emitter.get_message_viewers = AsyncMock(return_value=[])

    # Notes
    emitter.get_notes = AsyncMock(return_value=[])
    emitter.add_note = AsyncMock(return_value=MockNote())

    # Mentions
    emitter.extract_mentions = MagicMock(return_value=[])
    emitter.create_mention = AsyncMock()
    emitter.get_mentions_for_user = AsyncMock(return_value=[])
    emitter.acknowledge_mention = AsyncMock(return_value=True)

    # Activity
    emitter.emit_activity = AsyncMock()

    return emitter


@pytest.fixture(autouse=True)
def _patch_emitter(mock_emitter):
    """Patch the emitter singleton for every test."""
    with patch(
        "aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance",
        return_value=mock_emitter,
    ):
        yield


@pytest.fixture(autouse=True)
def _patch_activity_store():
    """Patch the activity store to avoid real I/O."""
    mock_store = MagicMock()
    mock_store.log_activity = MagicMock()
    with patch(
        "aragora.server.handlers.inbox.team_inbox._get_activity_store",
        return_value=mock_store,
    ):
        yield mock_store


# ===========================================================================
# Registration helper
# ===========================================================================


class TestGetTeamInboxHandlers:
    """Tests for get_team_inbox_handlers()."""

    def test_returns_dict(self):
        handlers = get_team_inbox_handlers()
        assert isinstance(handlers, dict)

    def test_contains_all_12_handlers(self):
        handlers = get_team_inbox_handlers()
        expected_keys = {
            "get_team_members",
            "add_team_member",
            "remove_team_member",
            "start_viewing",
            "stop_viewing",
            "start_typing",
            "stop_typing",
            "get_notes",
            "add_note",
            "get_mentions",
            "acknowledge_mention",
            "get_activity_feed",
        }
        assert set(handlers.keys()) == expected_keys

    def test_all_handlers_are_callable(self):
        handlers = get_team_inbox_handlers()
        for name, fn in handlers.items():
            assert callable(fn), f"Handler {name} is not callable"

    def test_handler_count(self):
        handlers = get_team_inbox_handlers()
        assert len(handlers) == 12


# ===========================================================================
# Get Team Members
# ===========================================================================


class TestHandleGetTeamMembers:
    """Tests for handle_get_team_members."""

    @pytest.mark.asyncio
    async def test_success_empty(self, mock_emitter):
        result = await handle_get_team_members(data={}, inbox_id="inbox-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["inbox_id"] == "inbox-1"
        assert body["data"]["team_members"] == []
        assert body["data"]["count"] == 0

    @pytest.mark.asyncio
    async def test_success_with_members(self, mock_emitter):
        members = [
            MockTeamMember("u1", "u1@example.com", "Alice", "admin"),
            MockTeamMember("u2", "u2@example.com", "Bob", "member"),
        ]
        mock_emitter.get_team_members.return_value = members
        result = await handle_get_team_members(data={}, inbox_id="inbox-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["count"] == 2
        assert len(body["data"]["team_members"]) == 2
        assert body["data"]["team_members"][0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        result = await handle_get_team_members(data={}, inbox_id="")
        assert _status(result) == 400
        assert "inbox_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_inbox_id_from_data(self, mock_emitter):
        result = await handle_get_team_members(data={"inbox_id": "inbox-2"}, inbox_id="")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["inbox_id"] == "inbox-2"
        mock_emitter.get_team_members.assert_awaited_once_with("inbox-2")

    @pytest.mark.asyncio
    async def test_inbox_id_param_takes_precedence(self, mock_emitter):
        result = await handle_get_team_members(
            data={"inbox_id": "inbox-data"}, inbox_id="inbox-param"
        )
        assert _status(result) == 200
        mock_emitter.get_team_members.assert_awaited_once_with("inbox-param")

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.get_team_members.side_effect = RuntimeError("db down")
        result = await handle_get_team_members(data={}, inbox_id="inbox-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, mock_emitter):
        mock_emitter.get_team_members.side_effect = ValueError("bad value")
        result = await handle_get_team_members(data={}, inbox_id="inbox-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_key_error_returns_500(self, mock_emitter):
        mock_emitter.get_team_members.side_effect = KeyError("missing")
        result = await handle_get_team_members(data={}, inbox_id="inbox-1")
        assert _status(result) == 500


# ===========================================================================
# Add Team Member
# ===========================================================================


class TestHandleAddTeamMember:
    """Tests for handle_add_team_member."""

    @pytest.mark.asyncio
    async def test_success(self, mock_emitter):
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New User",
            "role": "member",
        }
        result = await handle_add_team_member(data=data, inbox_id="inbox-1", user_id="admin-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "member" in body["data"]
        assert "Added" in body["data"]["message"]

    @pytest.mark.asyncio
    async def test_success_with_admin_role(self, mock_emitter):
        member = MockTeamMember("new-user", "new@example.com", "New Admin", "admin")
        mock_emitter.add_team_member.return_value = member
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New Admin",
            "role": "admin",
        }
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["member"]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_success_with_viewer_role(self, mock_emitter):
        member = MockTeamMember("new-user", "new@example.com", "Viewer", "viewer")
        mock_emitter.add_team_member.return_value = member
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "Viewer",
            "role": "viewer",
        }
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_default_role_is_member(self, mock_emitter):
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New User",
        }
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 200
        call_kwargs = mock_emitter.add_team_member.call_args
        assert call_kwargs.kwargs.get("role") == "member"

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        data = {"user_id": "u1", "email": "a@b.com", "name": "User"}
        result = await handle_add_team_member(data=data, inbox_id="")
        assert _status(result) == 400
        assert "inbox_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_inbox_id_from_data(self, mock_emitter):
        data = {
            "inbox_id": "inbox-data",
            "user_id": "u1",
            "email": "a@b.com",
            "name": "User",
        }
        result = await handle_add_team_member(data=data, inbox_id="")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_missing_user_id(self):
        data = {"email": "a@b.com", "name": "User"}
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 400
        assert "user_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_email(self):
        data = {"user_id": "u1", "name": "User"}
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 400
        assert "email" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_name(self):
        data = {"user_id": "u1", "email": "a@b.com"}
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 400
        assert "name" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_invalid_role(self):
        data = {"user_id": "u1", "email": "a@b.com", "name": "User", "role": "superadmin"}
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 400
        assert "role" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_emits_activity(self, mock_emitter):
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New User",
        }
        await handle_add_team_member(data=data, inbox_id="inbox-1", user_id="admin-1")
        mock_emitter.emit_activity.assert_awaited_once()
        call_kwargs = mock_emitter.emit_activity.call_args.kwargs
        assert call_kwargs["activity_type"] == "team_member_added"
        assert call_kwargs["inbox_id"] == "inbox-1"

    @pytest.mark.asyncio
    async def test_logs_activity_with_org_id(self, mock_emitter, _patch_activity_store):
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New User",
            "org_id": "org-1",
        }
        await handle_add_team_member(data=data, inbox_id="inbox-1", user_id="admin-1")
        _patch_activity_store.log_activity.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_activity_log_without_org_id(self, mock_emitter, _patch_activity_store):
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New User",
        }
        await handle_add_team_member(data=data, inbox_id="inbox-1")
        _patch_activity_store.log_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.add_team_member.side_effect = RuntimeError("fail")
        data = {"user_id": "u1", "email": "a@b.com", "name": "User"}
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_attribute_error_returns_500(self, mock_emitter):
        mock_emitter.add_team_member.side_effect = AttributeError("no attr")
        data = {"user_id": "u1", "email": "a@b.com", "name": "User"}
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        assert _status(result) == 500


# ===========================================================================
# Remove Team Member
# ===========================================================================


class TestHandleRemoveTeamMember:
    """Tests for handle_remove_team_member."""

    @pytest.mark.asyncio
    async def test_success(self, mock_emitter):
        result = await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id="user-2", user_id="admin-1"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["removed_user_id"] == "user-2"

    @pytest.mark.asyncio
    async def test_not_found(self, mock_emitter):
        mock_emitter.remove_team_member.return_value = False
        result = await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id="nonexistent"
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        result = await handle_remove_team_member(
            data={}, inbox_id="", member_user_id="user-2"
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_member_user_id(self):
        result = await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id=""
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_both_ids_missing(self):
        result = await handle_remove_team_member(data={}, inbox_id="", member_user_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_ids_from_data(self, mock_emitter):
        result = await handle_remove_team_member(
            data={"inbox_id": "inbox-data", "member_user_id": "user-data"},
            inbox_id="",
            member_user_id="",
        )
        assert _status(result) == 200
        mock_emitter.remove_team_member.assert_awaited_once_with("inbox-data", "user-data")

    @pytest.mark.asyncio
    async def test_emits_activity(self, mock_emitter):
        await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id="user-2", user_id="admin-1"
        )
        mock_emitter.emit_activity.assert_awaited_once()
        call_kwargs = mock_emitter.emit_activity.call_args.kwargs
        assert call_kwargs["activity_type"] == "team_member_removed"

    @pytest.mark.asyncio
    async def test_logs_activity_with_org_id(self, mock_emitter, _patch_activity_store):
        await handle_remove_team_member(
            data={"org_id": "org-1"},
            inbox_id="inbox-1",
            member_user_id="user-2",
        )
        _patch_activity_store.log_activity.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_activity_log_without_org_id(self, mock_emitter, _patch_activity_store):
        await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id="user-2"
        )
        _patch_activity_store.log_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.remove_team_member.side_effect = RuntimeError("fail")
        result = await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id="user-2"
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_no_emit_on_not_found(self, mock_emitter):
        mock_emitter.remove_team_member.return_value = False
        await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id="nonexistent"
        )
        mock_emitter.emit_activity.assert_not_awaited()


# ===========================================================================
# Start Viewing
# ===========================================================================


class TestHandleStartViewing:
    """Tests for handle_start_viewing."""

    @pytest.mark.asyncio
    async def test_success(self, mock_emitter):
        mock_emitter.get_message_viewers.return_value = ["user-1"]
        result = await handle_start_viewing(
            data={"user_name": "Alice"},
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-1",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["viewing"] is True
        assert body["data"]["message_id"] == "msg-1"
        assert body["data"]["current_viewers"] == ["user-1"]

    @pytest.mark.asyncio
    async def test_default_user_name(self, mock_emitter):
        await handle_start_viewing(
            data={}, inbox_id="inbox-1", message_id="msg-1", user_id="u1"
        )
        call_kwargs = mock_emitter.emit_user_viewing.call_args.kwargs
        assert call_kwargs["user_name"] == "Unknown"

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        result = await handle_start_viewing(
            data={}, inbox_id="", message_id="msg-1"
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_message_id(self):
        result = await handle_start_viewing(
            data={}, inbox_id="inbox-1", message_id=""
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_both_ids_missing(self):
        result = await handle_start_viewing(data={}, inbox_id="", message_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_ids_from_data(self, mock_emitter):
        result = await handle_start_viewing(
            data={"inbox_id": "inbox-data", "message_id": "msg-data"},
            inbox_id="",
            message_id="",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.emit_user_viewing.side_effect = RuntimeError("ws fail")
        result = await handle_start_viewing(
            data={}, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_emits_viewing_event(self, mock_emitter):
        await handle_start_viewing(
            data={"user_name": "Alice"},
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-1",
        )
        mock_emitter.emit_user_viewing.assert_awaited_once_with(
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-1",
            user_name="Alice",
        )


# ===========================================================================
# Stop Viewing
# ===========================================================================


class TestHandleStopViewing:
    """Tests for handle_stop_viewing."""

    @pytest.mark.asyncio
    async def test_success(self, mock_emitter):
        result = await handle_stop_viewing(
            data={"user_name": "Alice"},
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-1",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["viewing"] is False
        assert body["data"]["message_id"] == "msg-1"

    @pytest.mark.asyncio
    async def test_default_user_name(self, mock_emitter):
        await handle_stop_viewing(
            data={}, inbox_id="inbox-1", message_id="msg-1", user_id="u1"
        )
        call_kwargs = mock_emitter.emit_user_left.call_args.kwargs
        assert call_kwargs["user_name"] == "Unknown"

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        result = await handle_stop_viewing(data={}, inbox_id="", message_id="msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_message_id(self):
        result = await handle_stop_viewing(data={}, inbox_id="inbox-1", message_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_ids_from_data(self, mock_emitter):
        result = await handle_stop_viewing(
            data={"inbox_id": "inbox-d", "message_id": "msg-d"},
            inbox_id="",
            message_id="",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.emit_user_left.side_effect = ValueError("err")
        result = await handle_stop_viewing(
            data={}, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_emits_left_event(self, mock_emitter):
        await handle_stop_viewing(
            data={"user_name": "Bob"},
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-2",
        )
        mock_emitter.emit_user_left.assert_awaited_once_with(
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-2",
            user_name="Bob",
        )


# ===========================================================================
# Start Typing
# ===========================================================================


class TestHandleStartTyping:
    """Tests for handle_start_typing."""

    @pytest.mark.asyncio
    async def test_success(self, mock_emitter):
        result = await handle_start_typing(
            data={"user_name": "Alice"},
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-1",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["typing"] is True
        assert body["data"]["message_id"] == "msg-1"

    @pytest.mark.asyncio
    async def test_default_user_name(self, mock_emitter):
        await handle_start_typing(
            data={}, inbox_id="inbox-1", message_id="msg-1", user_id="u1"
        )
        call_kwargs = mock_emitter.emit_user_typing.call_args.kwargs
        assert call_kwargs["user_name"] == "Unknown"

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        result = await handle_start_typing(data={}, inbox_id="", message_id="msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_message_id(self):
        result = await handle_start_typing(data={}, inbox_id="inbox-1", message_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_both_ids_missing(self):
        result = await handle_start_typing(data={}, inbox_id="", message_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_ids_from_data(self, mock_emitter):
        result = await handle_start_typing(
            data={"inbox_id": "inbox-d", "message_id": "msg-d"},
            inbox_id="",
            message_id="",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.emit_user_typing.side_effect = TypeError("bad")
        result = await handle_start_typing(
            data={}, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_emits_typing_event(self, mock_emitter):
        await handle_start_typing(
            data={"user_name": "Carol"},
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-3",
        )
        mock_emitter.emit_user_typing.assert_awaited_once_with(
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-3",
            user_name="Carol",
        )


# ===========================================================================
# Stop Typing
# ===========================================================================


class TestHandleStopTyping:
    """Tests for handle_stop_typing."""

    @pytest.mark.asyncio
    async def test_success(self, mock_emitter):
        result = await handle_stop_typing(
            data={"user_name": "Alice"},
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-1",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["typing"] is False
        assert body["data"]["message_id"] == "msg-1"

    @pytest.mark.asyncio
    async def test_default_user_name(self, mock_emitter):
        await handle_stop_typing(
            data={}, inbox_id="inbox-1", message_id="msg-1", user_id="u1"
        )
        call_kwargs = mock_emitter.emit_user_stopped_typing.call_args.kwargs
        assert call_kwargs["user_name"] == "Unknown"

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        result = await handle_stop_typing(data={}, inbox_id="", message_id="msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_message_id(self):
        result = await handle_stop_typing(data={}, inbox_id="inbox-1", message_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_ids_from_data(self, mock_emitter):
        result = await handle_stop_typing(
            data={"inbox_id": "inbox-d", "message_id": "msg-d"},
            inbox_id="",
            message_id="",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.emit_user_stopped_typing.side_effect = AttributeError("no attr")
        result = await handle_stop_typing(
            data={}, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_emits_stopped_typing_event(self, mock_emitter):
        await handle_stop_typing(
            data={"user_name": "Dave"},
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-4",
        )
        mock_emitter.emit_user_stopped_typing.assert_awaited_once_with(
            inbox_id="inbox-1",
            message_id="msg-1",
            user_id="user-4",
            user_name="Dave",
        )


# ===========================================================================
# Get Notes
# ===========================================================================


class TestHandleGetNotes:
    """Tests for handle_get_notes."""

    @pytest.mark.asyncio
    async def test_success_empty(self, mock_emitter):
        result = await handle_get_notes(
            data={}, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["notes"] == []
        assert body["data"]["count"] == 0
        assert body["data"]["message_id"] == "msg-1"

    @pytest.mark.asyncio
    async def test_success_with_notes(self, mock_emitter):
        notes = [
            MockNote("n1", "First note", "u1", "Alice"),
            MockNote("n2", "Second note", "u2", "Bob"),
        ]
        mock_emitter.get_notes.return_value = notes
        result = await handle_get_notes(
            data={}, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["count"] == 2
        assert len(body["data"]["notes"]) == 2

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        result = await handle_get_notes(data={}, inbox_id="", message_id="msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_message_id(self):
        result = await handle_get_notes(data={}, inbox_id="inbox-1", message_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_both_ids_missing(self):
        result = await handle_get_notes(data={}, inbox_id="", message_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_ids_from_data(self, mock_emitter):
        result = await handle_get_notes(
            data={"inbox_id": "inbox-d", "message_id": "msg-d"},
            inbox_id="",
            message_id="",
        )
        assert _status(result) == 200
        mock_emitter.get_notes.assert_awaited_once_with("msg-d")

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.get_notes.side_effect = RuntimeError("db fail")
        result = await handle_get_notes(
            data={}, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 500


# ===========================================================================
# Add Note
# ===========================================================================


class TestHandleAddNote:
    """Tests for handle_add_note."""

    @pytest.mark.asyncio
    async def test_success(self, mock_emitter):
        data = {"content": "This is a note", "author_name": "Alice"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1", user_id="user-1"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["message"] == "Note added successfully"
        assert "note" in body["data"]

    @pytest.mark.asyncio
    async def test_default_author_name(self, mock_emitter):
        data = {"content": "A note"}
        await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1", user_id="u1"
        )
        call_kwargs = mock_emitter.add_note.call_args.kwargs
        assert call_kwargs["author_name"] == "Unknown"

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        data = {"content": "A note"}
        result = await handle_add_note(data=data, inbox_id="", message_id="msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_message_id(self):
        data = {"content": "A note"}
        result = await handle_add_note(data=data, inbox_id="inbox-1", message_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_content(self):
        data = {"author_name": "Alice"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 400
        assert "content" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_empty_content(self):
        data = {"content": "", "author_name": "Alice"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_with_mentions(self, mock_emitter):
        mock_emitter.extract_mentions.return_value = ["alice", "bob"]
        data = {"content": "Hey @alice and @bob, check this out"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1", user_id="user-1"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["mentions_created"] == 2
        assert mock_emitter.create_mention.await_count == 2

    @pytest.mark.asyncio
    async def test_no_mentions(self, mock_emitter):
        mock_emitter.extract_mentions.return_value = []
        data = {"content": "Just a regular note"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["mentions_created"] == 0
        mock_emitter.create_mention.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mention_creation_failure_does_not_break(self, mock_emitter):
        mock_emitter.extract_mentions.return_value = ["alice", "bob"]
        mock_emitter.create_mention.side_effect = ValueError("user not found")
        data = {"content": "Hey @alice @bob"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        # Should still succeed even if mention creation fails
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_logs_activity_with_org_id(self, mock_emitter, _patch_activity_store):
        data = {"content": "A note", "org_id": "org-1"}
        await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        _patch_activity_store.log_activity.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_activity_log_without_org_id(self, mock_emitter, _patch_activity_store):
        data = {"content": "A note"}
        await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        _patch_activity_store.log_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.add_note.side_effect = RuntimeError("fail")
        data = {"content": "A note"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_ids_from_data(self, mock_emitter):
        data = {
            "inbox_id": "inbox-d",
            "message_id": "msg-d",
            "content": "A note",
        }
        result = await handle_add_note(data=data, inbox_id="", message_id="")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_note_to_dict_in_response(self, mock_emitter):
        note = MockNote("note-42", "Hello world", "u1", "Alice")
        mock_emitter.add_note.return_value = note
        data = {"content": "Hello world"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        body = _body(result)
        assert body["data"]["note"]["id"] == "note-42"
        assert body["data"]["note"]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_mention_context_truncated(self, mock_emitter):
        """Verify that mention context is truncated to first 100 chars."""
        long_content = "x" * 200
        mock_emitter.extract_mentions.return_value = ["alice"]
        data = {"content": long_content}
        await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1", user_id="user-1"
        )
        call_kwargs = mock_emitter.create_mention.call_args.kwargs
        assert len(call_kwargs["context"]) == 100

    @pytest.mark.asyncio
    async def test_mention_runtime_error_continues(self, mock_emitter):
        """RuntimeError during mention creation is caught and logged."""
        mock_emitter.extract_mentions.return_value = ["alice"]
        mock_emitter.create_mention.side_effect = RuntimeError("boom")
        data = {"content": "Hey @alice"}
        result = await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1"
        )
        assert _status(result) == 200


# ===========================================================================
# Get Mentions
# ===========================================================================


class TestHandleGetMentions:
    """Tests for handle_get_mentions."""

    @pytest.mark.asyncio
    async def test_success_empty(self, mock_emitter):
        result = await handle_get_mentions(data={}, user_id="user-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["mentions"] == []
        assert body["data"]["count"] == 0
        assert body["data"]["unacknowledged_count"] == 0

    @pytest.mark.asyncio
    async def test_success_with_mentions(self, mock_emitter):
        mentions = [
            MockMention("m1", "user-1", False),
            MockMention("m2", "user-1", True),
            MockMention("m3", "user-1", False),
        ]
        mock_emitter.get_mentions_for_user.return_value = mentions
        result = await handle_get_mentions(data={}, user_id="user-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["count"] == 3
        assert body["data"]["unacknowledged_count"] == 2

    @pytest.mark.asyncio
    async def test_unacknowledged_only_bool(self, mock_emitter):
        await handle_get_mentions(data={"unacknowledged_only": True}, user_id="user-1")
        call_kwargs = mock_emitter.get_mentions_for_user.call_args.kwargs
        assert call_kwargs["unacknowledged_only"] is True

    @pytest.mark.asyncio
    async def test_unacknowledged_only_string_true(self, mock_emitter):
        await handle_get_mentions(data={"unacknowledged_only": "true"}, user_id="user-1")
        call_kwargs = mock_emitter.get_mentions_for_user.call_args.kwargs
        assert call_kwargs["unacknowledged_only"] is True

    @pytest.mark.asyncio
    async def test_unacknowledged_only_string_false(self, mock_emitter):
        await handle_get_mentions(data={"unacknowledged_only": "false"}, user_id="user-1")
        call_kwargs = mock_emitter.get_mentions_for_user.call_args.kwargs
        assert call_kwargs["unacknowledged_only"] is False

    @pytest.mark.asyncio
    async def test_unacknowledged_only_string_TRUE(self, mock_emitter):
        await handle_get_mentions(data={"unacknowledged_only": "TRUE"}, user_id="user-1")
        call_kwargs = mock_emitter.get_mentions_for_user.call_args.kwargs
        assert call_kwargs["unacknowledged_only"] is True

    @pytest.mark.asyncio
    async def test_default_unacknowledged_only(self, mock_emitter):
        await handle_get_mentions(data={}, user_id="user-1")
        call_kwargs = mock_emitter.get_mentions_for_user.call_args.kwargs
        assert call_kwargs["unacknowledged_only"] is False

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.get_mentions_for_user.side_effect = RuntimeError("fail")
        result = await handle_get_mentions(data={}, user_id="user-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_all_acknowledged(self, mock_emitter):
        mentions = [
            MockMention("m1", "user-1", True),
            MockMention("m2", "user-1", True),
        ]
        mock_emitter.get_mentions_for_user.return_value = mentions
        result = await handle_get_mentions(data={}, user_id="user-1")
        body = _body(result)
        assert body["data"]["unacknowledged_count"] == 0

    @pytest.mark.asyncio
    async def test_type_error_returns_500(self, mock_emitter):
        mock_emitter.get_mentions_for_user.side_effect = TypeError("bad type")
        result = await handle_get_mentions(data={}, user_id="user-1")
        assert _status(result) == 500


# ===========================================================================
# Acknowledge Mention
# ===========================================================================


class TestHandleAcknowledgeMention:
    """Tests for handle_acknowledge_mention."""

    @pytest.mark.asyncio
    async def test_success(self, mock_emitter):
        result = await handle_acknowledge_mention(
            data={}, mention_id="mention-1", user_id="user-1"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["acknowledged"] is True
        assert body["data"]["mention_id"] == "mention-1"
        assert "acknowledged_at" in body["data"]

    @pytest.mark.asyncio
    async def test_not_found(self, mock_emitter):
        mock_emitter.acknowledge_mention.return_value = False
        result = await handle_acknowledge_mention(
            data={}, mention_id="nonexistent", user_id="user-1"
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_mention_id(self):
        result = await handle_acknowledge_mention(data={}, mention_id="")
        assert _status(result) == 400
        assert "mention_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_mention_id_from_data(self, mock_emitter):
        result = await handle_acknowledge_mention(
            data={"mention_id": "mention-data"}, mention_id=""
        )
        assert _status(result) == 200
        mock_emitter.acknowledge_mention.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mention_id_param_takes_precedence(self, mock_emitter):
        result = await handle_acknowledge_mention(
            data={"mention_id": "mention-data"}, mention_id="mention-param"
        )
        assert _status(result) == 200
        call_args = mock_emitter.acknowledge_mention.call_args
        assert call_args[0][1] == "mention-param" or call_args.kwargs.get("mention_id") == "mention-param" or call_args[0] == ("default", "mention-param")

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, mock_emitter):
        mock_emitter.acknowledge_mention.side_effect = RuntimeError("fail")
        result = await handle_acknowledge_mention(
            data={}, mention_id="mention-1"
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_acknowledged_at_is_iso_format(self, mock_emitter):
        result = await handle_acknowledge_mention(
            data={}, mention_id="mention-1", user_id="user-1"
        )
        body = _body(result)
        ack_at = body["data"]["acknowledged_at"]
        # Should parse without error
        datetime.fromisoformat(ack_at)

    @pytest.mark.asyncio
    async def test_key_error_returns_500(self, mock_emitter):
        mock_emitter.acknowledge_mention.side_effect = KeyError("missing key")
        result = await handle_acknowledge_mention(
            data={}, mention_id="mention-1"
        )
        assert _status(result) == 500


# ===========================================================================
# Activity Feed
# ===========================================================================


class TestHandleGetActivityFeed:
    """Tests for handle_get_activity_feed."""

    @pytest.mark.asyncio
    async def test_success(self):
        result = await handle_get_activity_feed(data={}, inbox_id="inbox-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["inbox_id"] == "inbox-1"
        assert body["data"]["activities"] == []
        assert "message" in body["data"]

    @pytest.mark.asyncio
    async def test_missing_inbox_id(self):
        result = await handle_get_activity_feed(data={}, inbox_id="")
        assert _status(result) == 400
        assert "inbox_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_inbox_id_from_data(self):
        result = await handle_get_activity_feed(
            data={"inbox_id": "inbox-data"}, inbox_id=""
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["inbox_id"] == "inbox-data"

    @pytest.mark.asyncio
    async def test_default_limit(self):
        result = await handle_get_activity_feed(data={}, inbox_id="inbox-1")
        body = _body(result)
        assert body["data"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_custom_limit(self):
        result = await handle_get_activity_feed(
            data={"limit": 25}, inbox_id="inbox-1"
        )
        body = _body(result)
        assert body["data"]["limit"] == 25

    @pytest.mark.asyncio
    async def test_limit_capped_at_200(self):
        result = await handle_get_activity_feed(
            data={"limit": 500}, inbox_id="inbox-1"
        )
        body = _body(result)
        assert body["data"]["limit"] == 200

    @pytest.mark.asyncio
    async def test_limit_minimum_1(self):
        result = await handle_get_activity_feed(
            data={"limit": 0}, inbox_id="inbox-1"
        )
        body = _body(result)
        assert body["data"]["limit"] == 1

    @pytest.mark.asyncio
    async def test_negative_limit_clamped(self):
        result = await handle_get_activity_feed(
            data={"limit": -10}, inbox_id="inbox-1"
        )
        body = _body(result)
        assert body["data"]["limit"] == 1

    @pytest.mark.asyncio
    async def test_default_offset(self):
        result = await handle_get_activity_feed(data={}, inbox_id="inbox-1")
        body = _body(result)
        assert body["data"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_custom_offset(self):
        result = await handle_get_activity_feed(
            data={"offset": 10}, inbox_id="inbox-1"
        )
        body = _body(result)
        assert body["data"]["offset"] == 10

    @pytest.mark.asyncio
    async def test_negative_offset_clamped(self):
        result = await handle_get_activity_feed(
            data={"offset": -5}, inbox_id="inbox-1"
        )
        body = _body(result)
        assert body["data"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_invalid_limit_returns_500(self):
        result = await handle_get_activity_feed(
            data={"limit": "not_a_number"}, inbox_id="inbox-1"
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_websocket_message_present(self):
        result = await handle_get_activity_feed(data={}, inbox_id="inbox-1")
        body = _body(result)
        assert "websocket" in body["data"]["message"].lower()


# ===========================================================================
# Activity Logging Helper
# ===========================================================================


class TestLogActivity:
    """Tests for the _log_activity helper."""

    @pytest.mark.asyncio
    async def test_activity_logged_on_add_member(self, mock_emitter, _patch_activity_store):
        """Verify _log_activity is called correctly when adding a team member with org_id."""
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New User",
            "org_id": "org-1",
        }
        await handle_add_team_member(data=data, inbox_id="inbox-1", user_id="admin-1")
        _patch_activity_store.log_activity.assert_called_once()

    @pytest.mark.asyncio
    async def test_activity_logged_on_remove_member(self, mock_emitter, _patch_activity_store):
        """Verify _log_activity is called when removing a team member with org_id."""
        await handle_remove_team_member(
            data={"org_id": "org-1"},
            inbox_id="inbox-1",
            member_user_id="user-2",
            user_id="admin-1",
        )
        _patch_activity_store.log_activity.assert_called_once()

    @pytest.mark.asyncio
    async def test_activity_logged_on_add_note(self, mock_emitter, _patch_activity_store):
        """Verify _log_activity is called when adding a note with org_id."""
        data = {"content": "A note", "org_id": "org-1"}
        await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1", user_id="user-1"
        )
        _patch_activity_store.log_activity.assert_called_once()

    def test_log_activity_no_store(self):
        """When store is None, _log_activity should not raise."""
        from aragora.server.handlers.inbox.team_inbox import _log_activity

        with patch(
            "aragora.server.handlers.inbox.team_inbox._get_activity_store",
            return_value=None,
        ):
            # Should not raise
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="test",
            )

    def test_log_activity_store_error_swallowed(self):
        """When store.log_activity raises, it should be caught silently."""
        from aragora.server.handlers.inbox.team_inbox import _log_activity

        mock_store = MagicMock()
        mock_store.log_activity.side_effect = OSError("disk full")
        with patch(
            "aragora.server.handlers.inbox.team_inbox._get_activity_store",
            return_value=mock_store,
        ):
            # Should not raise even when log_activity raises
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="test",
            )


# ===========================================================================
# Edge Cases and Cross-Cutting Concerns
# ===========================================================================


class TestEdgeCases:
    """Cross-cutting edge cases and boundary tests."""

    @pytest.mark.asyncio
    async def test_get_team_members_returns_correct_member_dicts(self, mock_emitter):
        member = MockTeamMember("u1", "alice@example.com", "Alice", "admin")
        mock_emitter.get_team_members.return_value = [member]
        result = await handle_get_team_members(data={}, inbox_id="inbox-1")
        body = _body(result)
        team_members = body["data"]["team_members"]
        assert team_members[0]["user_id"] == "u1"
        assert team_members[0]["email"] == "alice@example.com"
        assert team_members[0]["name"] == "Alice"
        assert team_members[0]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_add_member_passes_correct_args(self, mock_emitter):
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New User",
            "role": "viewer",
        }
        await handle_add_team_member(data=data, inbox_id="inbox-1", user_id="admin-1")
        mock_emitter.add_team_member.assert_awaited_once_with(
            inbox_id="inbox-1",
            user_id="new-user",
            email="new@example.com",
            name="New User",
            role="viewer",
        )

    @pytest.mark.asyncio
    async def test_add_note_passes_correct_args(self, mock_emitter):
        data = {"content": "Hello", "author_name": "Alice"}
        await handle_add_note(
            data=data, inbox_id="inbox-1", message_id="msg-1", user_id="user-1"
        )
        mock_emitter.add_note.assert_awaited_once_with(
            inbox_id="inbox-1",
            message_id="msg-1",
            author_id="user-1",
            author_name="Alice",
            content="Hello",
        )

    @pytest.mark.asyncio
    async def test_acknowledge_passes_user_and_mention(self, mock_emitter):
        await handle_acknowledge_mention(
            data={}, mention_id="m-1", user_id="user-1"
        )
        mock_emitter.acknowledge_mention.assert_awaited_once_with("user-1", "m-1")

    @pytest.mark.asyncio
    async def test_remove_member_passes_correct_args(self, mock_emitter):
        await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id="target-user"
        )
        mock_emitter.remove_team_member.assert_awaited_once_with("inbox-1", "target-user")

    @pytest.mark.asyncio
    async def test_all_valid_roles_accepted(self, mock_emitter):
        for role in ("admin", "member", "viewer"):
            mock_emitter.add_team_member.return_value = MockTeamMember(role=role)
            data = {
                "user_id": "u1",
                "email": "a@b.com",
                "name": "User",
                "role": role,
            }
            result = await handle_add_team_member(data=data, inbox_id="inbox-1")
            assert _status(result) == 200, f"Role '{role}' should be valid"

    @pytest.mark.asyncio
    async def test_invalid_roles_rejected(self, mock_emitter):
        for role in ("owner", "moderator", "root", "", "ADMIN"):
            data = {
                "user_id": "u1",
                "email": "a@b.com",
                "name": "User",
                "role": role,
            }
            result = await handle_add_team_member(data=data, inbox_id="inbox-1")
            if role == "":
                # empty role falls through to validation since it's not in valid_roles
                # but data.get("role", "member") returns "" which is not in valid_roles
                assert _status(result) == 400, f"Empty role should be rejected"
            else:
                assert _status(result) == 400, f"Role '{role}' should be rejected"

    @pytest.mark.asyncio
    async def test_concurrent_viewers(self, mock_emitter):
        """Multiple viewers can be returned."""
        mock_emitter.get_message_viewers.return_value = ["u1", "u2", "u3"]
        result = await handle_start_viewing(
            data={}, inbox_id="inbox-1", message_id="msg-1"
        )
        body = _body(result)
        assert len(body["data"]["current_viewers"]) == 3

    @pytest.mark.asyncio
    async def test_add_member_emit_activity_metadata(self, mock_emitter):
        """Verify emit_activity metadata contains the added user info."""
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "New User",
            "role": "member",
        }
        await handle_add_team_member(data=data, inbox_id="inbox-1", user_id="admin-1")
        call_kwargs = mock_emitter.emit_activity.call_args.kwargs
        assert call_kwargs["metadata"]["addedUserId"] == "new-user"
        assert call_kwargs["metadata"]["role"] == "member"

    @pytest.mark.asyncio
    async def test_remove_member_emit_activity_metadata(self, mock_emitter):
        """Verify emit_activity metadata contains the removed user id."""
        await handle_remove_team_member(
            data={}, inbox_id="inbox-1", member_user_id="removed-user", user_id="admin-1"
        )
        call_kwargs = mock_emitter.emit_activity.call_args.kwargs
        assert call_kwargs["metadata"]["removedUserId"] == "removed-user"

    @pytest.mark.asyncio
    async def test_add_member_emit_description(self, mock_emitter):
        data = {
            "user_id": "new-user",
            "email": "new@example.com",
            "name": "Alice",
        }
        await handle_add_team_member(data=data, inbox_id="inbox-1")
        call_kwargs = mock_emitter.emit_activity.call_args.kwargs
        assert "Alice" in call_kwargs["description"]

    @pytest.mark.asyncio
    async def test_add_member_response_message(self, mock_emitter):
        mock_emitter.add_team_member.return_value = MockTeamMember(name="Alice")
        data = {
            "user_id": "u1",
            "email": "a@b.com",
            "name": "Alice",
        }
        result = await handle_add_team_member(data=data, inbox_id="inbox-1")
        body = _body(result)
        assert "Alice" in body["data"]["message"]
