"""Tests for SharedInboxHandler (aragora/server/handlers/shared_inbox/handler.py).

Covers all handler methods in the SharedInboxHandler class:
- can_handle                              Route matching
- handle                                  Sync dispatch (returns None)
- handle_post_shared_inbox                POST /api/v1/inbox/shared
- handle_get_shared_inboxes               GET  /api/v1/inbox/shared
- handle_get_shared_inbox                 GET  /api/v1/inbox/shared/:id
- handle_get_inbox_messages               GET  /api/v1/inbox/shared/:id/messages
- handle_post_assign_message              POST /api/v1/inbox/shared/:id/messages/:msg_id/assign
- handle_post_update_status               POST /api/v1/inbox/shared/:id/messages/:msg_id/status
- handle_post_add_tag                     POST /api/v1/inbox/shared/:id/messages/:msg_id/tag
- handle_post_routing_rule                POST /api/v1/inbox/routing/rules
- handle_get_routing_rules                GET  /api/v1/inbox/routing/rules
- handle_patch_routing_rule               PATCH /api/v1/inbox/routing/rules/:id
- handle_delete_routing_rule              DELETE /api/v1/inbox/routing/rules/:id
- handle_post_test_routing_rule           POST /api/v1/inbox/routing/rules/:id/test
- _get_user_id                            Auth helper
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.shared_inbox.handler import SharedInboxHandler
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
# Module-level patch paths
# ---------------------------------------------------------------------------

_HANDLER_MOD = "aragora.server.handlers.shared_inbox.handler"
_INBOX_HANDLERS_MOD = "aragora.server.handlers.shared_inbox.handler"
_RULE_HANDLERS_MOD = "aragora.server.handlers.shared_inbox.handler"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a SharedInboxHandler with a mock server context."""
    ctx: dict[str, Any] = {}
    return SharedInboxHandler(ctx)


@pytest.fixture
def handler_with_auth():
    """Create a SharedInboxHandler with auth context providing a user ID."""
    auth_ctx = MagicMock()
    auth_ctx.user_id = "user-42"
    ctx: dict[str, Any] = {"auth_context": auth_ctx}
    return SharedInboxHandler(ctx)


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for SharedInboxHandler.can_handle()."""

    def test_exact_shared_inbox_route(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox/shared") is True

    def test_exact_routing_rules_route(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox/routing/rules") is True

    def test_shared_inbox_prefix(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox/shared/inbox_abc123") is True

    def test_shared_inbox_messages_prefix(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox/shared/inbox_abc/messages") is True

    def test_routing_rules_prefix(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox/routing/rules/rule_abc") is True

    def test_routing_rules_test_prefix(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox/routing/rules/rule_abc/test") is True

    def test_unrelated_path_returns_false(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_partial_path_returns_false(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox") is False

    def test_empty_path_returns_false(self, handler: SharedInboxHandler):
        assert handler.can_handle("") is False

    def test_similar_but_wrong_prefix(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox/shared_extra") is False

    def test_inbox_routing_without_rules(self, handler: SharedInboxHandler):
        assert handler.can_handle("/api/v1/inbox/routing") is False


# ===========================================================================
# handle (sync dispatch)
# ===========================================================================


class TestHandle:
    """Tests for SharedInboxHandler.handle() sync method."""

    def test_returns_none(self, handler: SharedInboxHandler):
        result = handler.handle("/api/v1/inbox/shared", {}, MagicMock())
        assert result is None


# ===========================================================================
# _get_user_id
# ===========================================================================


class TestGetUserId:
    """Tests for SharedInboxHandler._get_user_id()."""

    def test_returns_user_id_from_auth_context(self, handler_with_auth: SharedInboxHandler):
        assert handler_with_auth._get_user_id() == "user-42"

    def test_returns_default_when_no_auth_context(self, handler: SharedInboxHandler):
        assert handler._get_user_id() == "default"

    def test_returns_default_when_auth_context_has_no_user_id(self):
        ctx: dict[str, Any] = {"auth_context": object()}
        h = SharedInboxHandler(ctx)
        assert h._get_user_id() == "default"

    def test_returns_default_when_auth_context_is_none(self):
        ctx: dict[str, Any] = {"auth_context": None}
        h = SharedInboxHandler(ctx)
        assert h._get_user_id() == "default"


# ===========================================================================
# handle_post_shared_inbox
# ===========================================================================


class TestPostSharedInbox:
    """Tests for POST /api/v1/inbox/shared."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "inbox": {"id": "inbox_1", "name": "Support"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "Support",
                    "description": "Support inbox",
                    "email_address": "support@example.com",
                }
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler: SharedInboxHandler):
        result = await handler.handle_post_shared_inbox({"name": "Test"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_name(self, handler: SharedInboxHandler):
        result = await handler.handle_post_shared_inbox({"workspace_id": "ws_1"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_both_fields(self, handler: SharedInboxHandler):
        result = await handler.handle_post_shared_inbox({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_inbox_name_too_long(self, handler: SharedInboxHandler):
        result = await handler.handle_post_shared_inbox(
            {
                "workspace_id": "ws_1",
                "name": "x" * 300,
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_email_address(self, handler: SharedInboxHandler):
        result = await handler.handle_post_shared_inbox(
            {
                "workspace_id": "ws_1",
                "name": "Test",
                "email_address": "not-an-email",
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handler_returns_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Duplicate name"}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "Support",
                }
            )
        assert _status(result) == 400
        body = _body(result)
        assert "Duplicate name" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_sanitizes_name(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "inbox": {"id": "inbox_1", "name": "Good Name"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "  Good Name  ",
                }
            )
            # The sanitized name should be passed
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["name"] is not None

    @pytest.mark.asyncio
    async def test_passes_optional_fields(self, handler_with_auth: SharedInboxHandler):
        mock_result = {"success": True, "inbox": {"id": "inbox_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "connector_type": "gmail",
                    "team_members": ["user1", "user2"],
                    "admins": ["admin1"],
                    "settings": {"auto_assign": True},
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["connector_type"] == "gmail"
            assert call_kwargs["team_members"] == ["user1", "user2"]
            assert call_kwargs["admins"] == ["admin1"]
            assert call_kwargs["settings"] == {"auto_assign": True}
            assert call_kwargs["created_by"] == "user-42"

    @pytest.mark.asyncio
    async def test_description_sanitized(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "inbox": {"id": "inbox_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "description": "A description",
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["description"] is not None

    @pytest.mark.asyncio
    async def test_error_with_unknown_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                }
            )
        assert _status(result) == 400
        body = _body(result)
        assert "Unknown error" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_description_too_long(self, handler: SharedInboxHandler):
        result = await handler.handle_post_shared_inbox(
            {
                "workspace_id": "ws_1",
                "name": "Test",
                "description": "d" * 1500,
            }
        )
        assert _status(result) == 400


# ===========================================================================
# handle_get_shared_inboxes
# ===========================================================================


class TestGetSharedInboxes:
    """Tests for GET /api/v1/inbox/shared."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "inboxes": [{"id": "inbox_1"}], "total": 1}
        with patch(
            f"{_HANDLER_MOD}.handle_list_shared_inboxes",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_shared_inboxes({"workspace_id": "ws_1"})
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler: SharedInboxHandler):
        result = await handler.handle_get_shared_inboxes({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handler_returns_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Access denied"}
        with patch(
            f"{_HANDLER_MOD}.handle_list_shared_inboxes",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_shared_inboxes({"workspace_id": "ws_1"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_passes_user_id(self, handler_with_auth: SharedInboxHandler):
        mock_result = {"success": True, "inboxes": [], "total": 0}
        with patch(
            f"{_HANDLER_MOD}.handle_list_shared_inboxes",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_get_shared_inboxes({"workspace_id": "ws_1"})
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["user_id"] == "user-42"


# ===========================================================================
# handle_get_shared_inbox
# ===========================================================================


class TestGetSharedInbox:
    """Tests for GET /api/v1/inbox/shared/:id."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "inbox": {"id": "inbox_1", "name": "Support"}}
        with patch(
            f"{_HANDLER_MOD}.handle_get_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_shared_inbox({}, "inbox_1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_not_found(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Inbox not found"}
        with patch(
            f"{_HANDLER_MOD}.handle_get_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_shared_inbox({}, "nonexistent")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_error_returns_404(self, handler: SharedInboxHandler):
        mock_result = {"success": False}
        with patch(
            f"{_HANDLER_MOD}.handle_get_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_shared_inbox({}, "inbox_1")
        assert _status(result) == 404


# ===========================================================================
# handle_get_inbox_messages
# ===========================================================================


class TestGetInboxMessages:
    """Tests for GET /api/v1/inbox/shared/:id/messages."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {
            "success": True,
            "messages": [{"id": "msg_1"}],
            "total": 1,
            "limit": 50,
            "offset": 0,
        }
        with patch(
            f"{_HANDLER_MOD}.handle_get_inbox_messages",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_inbox_messages({}, "inbox_1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_with_filters(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "messages": [], "total": 0, "limit": 10, "offset": 5}
        with patch(
            f"{_HANDLER_MOD}.handle_get_inbox_messages",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox_messages(
                {
                    "status": "open",
                    "assigned_to": "user1",
                    "tag": "urgent",
                    "limit": "10",
                    "offset": "5",
                },
                "inbox_1",
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["status"] == "open"
            assert call_kwargs["assigned_to"] == "user1"
            assert call_kwargs["tag"] == "urgent"
            assert call_kwargs["limit"] == 10
            assert call_kwargs["offset"] == 5

    @pytest.mark.asyncio
    async def test_default_limit_and_offset(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "messages": [], "total": 0, "limit": 50, "offset": 0}
        with patch(
            f"{_HANDLER_MOD}.handle_get_inbox_messages",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox_messages({}, "inbox_1")
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["limit"] == 50
            assert call_kwargs["offset"] == 0

    @pytest.mark.asyncio
    async def test_handler_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Inbox not found"}
        with patch(
            f"{_HANDLER_MOD}.handle_get_inbox_messages",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_inbox_messages({}, "inbox_1")
        assert _status(result) == 400


# ===========================================================================
# handle_post_assign_message
# ===========================================================================


class TestPostAssignMessage:
    """Tests for POST /api/v1/inbox/shared/:id/messages/:msg_id/assign."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "assigned_to": "user1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_assign_message",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_assign_message(
                {"assigned_to": "user1"}, "inbox_1", "msg_1"
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_missing_assigned_to(self, handler: SharedInboxHandler):
        result = await handler.handle_post_assign_message({}, "inbox_1", "msg_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_assigned_to_not_string(self, handler: SharedInboxHandler):
        result = await handler.handle_post_assign_message({"assigned_to": 123}, "inbox_1", "msg_1")
        assert _status(result) == 400
        body = _body(result)
        assert "string" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_assigned_to_too_long(self, handler: SharedInboxHandler):
        result = await handler.handle_post_assign_message(
            {"assigned_to": "x" * 300}, "inbox_1", "msg_1"
        )
        assert _status(result) == 400
        body = _body(result)
        assert "200" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_assigned_to_empty_after_sanitize(self, handler: SharedInboxHandler):
        # A string of only control characters should sanitize to empty
        with patch(f"{_HANDLER_MOD}.sanitize_user_input", return_value=""):
            result = await handler.handle_post_assign_message(
                {"assigned_to": "\x00\x01"}, "inbox_1", "msg_1"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handler_returns_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Message not found"}
        with patch(
            f"{_HANDLER_MOD}.handle_assign_message",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_assign_message(
                {"assigned_to": "user1"}, "inbox_1", "msg_1"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_passes_assigned_by(self, handler_with_auth: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_assign_message",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_assign_message(
                {"assigned_to": "user1"}, "inbox_1", "msg_1"
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["assigned_by"] == "user-42"


# ===========================================================================
# handle_post_update_status
# ===========================================================================


class TestPostUpdateStatus:
    """Tests for POST /api/v1/inbox/shared/:id/messages/:msg_id/status."""

    @pytest.mark.asyncio
    async def test_success_open(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "status": "open"}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_message_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_update_status({"status": "open"}, "inbox_1", "msg_1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_success_resolved(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "status": "resolved"}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_message_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_update_status(
                {"status": "resolved"}, "inbox_1", "msg_1"
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_success_assigned(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "status": "assigned"}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_message_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_update_status(
                {"status": "assigned"}, "inbox_1", "msg_1"
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_success_in_progress(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "status": "in_progress"}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_message_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_update_status(
                {"status": "in_progress"}, "inbox_1", "msg_1"
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_success_waiting(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "status": "waiting"}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_message_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_update_status(
                {"status": "waiting"}, "inbox_1", "msg_1"
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_success_closed(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "status": "closed"}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_message_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_update_status(
                {"status": "closed"}, "inbox_1", "msg_1"
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_missing_status(self, handler: SharedInboxHandler):
        result = await handler.handle_post_update_status({}, "inbox_1", "msg_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_status(self, handler: SharedInboxHandler):
        result = await handler.handle_post_update_status(
            {"status": "invalid_status"}, "inbox_1", "msg_1"
        )
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid status" in body.get("error", "")
        # Should list valid statuses
        assert "open" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_handler_returns_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Message not found"}
        with patch(
            f"{_HANDLER_MOD}.handle_update_message_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_update_status({"status": "open"}, "inbox_1", "msg_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_passes_updated_by(self, handler_with_auth: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_message_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_update_status(
                {"status": "open"}, "inbox_1", "msg_1"
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["updated_by"] == "user-42"


# ===========================================================================
# handle_post_add_tag
# ===========================================================================


class TestPostAddTag:
    """Tests for POST /api/v1/inbox/shared/:id/messages/:msg_id/tag."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "tags": ["urgent"]}}
        with patch(
            f"{_HANDLER_MOD}.handle_add_message_tag",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_add_tag({"tag": "urgent"}, "inbox_1", "msg_1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_missing_tag(self, handler: SharedInboxHandler):
        result = await handler.handle_post_add_tag({}, "inbox_1", "msg_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_tag(self, handler: SharedInboxHandler):
        result = await handler.handle_post_add_tag({"tag": ""}, "inbox_1", "msg_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_tag_too_long(self, handler: SharedInboxHandler):
        result = await handler.handle_post_add_tag({"tag": "x" * 200}, "inbox_1", "msg_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_tag_invalid_characters(self, handler: SharedInboxHandler):
        result = await handler.handle_post_add_tag({"tag": "has spaces!"}, "inbox_1", "msg_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_tag_with_hyphen_and_underscore(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "tags": ["my-tag_1"]}}
        with patch(
            f"{_HANDLER_MOD}.handle_add_message_tag",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_add_tag({"tag": "my-tag_1"}, "inbox_1", "msg_1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handler_returns_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Message not found"}
        with patch(
            f"{_HANDLER_MOD}.handle_add_message_tag",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_add_tag({"tag": "urgent"}, "inbox_1", "msg_1")
        assert _status(result) == 400


# ===========================================================================
# handle_post_routing_rule
# ===========================================================================


class TestPostRoutingRule:
    """Tests for POST /api/v1/inbox/routing/rules."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rule": {"id": "rule_1", "name": "Test Rule"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_routing_rule(
                {
                    "workspace_id": "ws_1",
                    "name": "Test Rule",
                    "conditions": [{"field": "subject", "operator": "contains", "value": "urgent"}],
                    "actions": [{"type": "assign", "target": "team1"}],
                }
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler: SharedInboxHandler):
        result = await handler.handle_post_routing_rule(
            {
                "name": "Test",
                "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
                "actions": [{"type": "assign", "target": "team1"}],
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_name(self, handler: SharedInboxHandler):
        result = await handler.handle_post_routing_rule(
            {
                "workspace_id": "ws_1",
                "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
                "actions": [{"type": "assign", "target": "team1"}],
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_conditions(self, handler: SharedInboxHandler):
        result = await handler.handle_post_routing_rule(
            {
                "workspace_id": "ws_1",
                "name": "Test",
                "actions": [{"type": "assign", "target": "team1"}],
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_actions(self, handler: SharedInboxHandler):
        result = await handler.handle_post_routing_rule(
            {
                "workspace_id": "ws_1",
                "name": "Test",
                "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_conditions(self, handler: SharedInboxHandler):
        result = await handler.handle_post_routing_rule(
            {
                "workspace_id": "ws_1",
                "name": "Test",
                "conditions": [],
                "actions": [{"type": "assign", "target": "team1"}],
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_actions(self, handler: SharedInboxHandler):
        result = await handler.handle_post_routing_rule(
            {
                "workspace_id": "ws_1",
                "name": "Test",
                "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
                "actions": [],
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_passes_optional_fields(self, handler_with_auth: SharedInboxHandler):
        mock_result = {"success": True, "rule": {"id": "rule_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_routing_rule(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
                    "actions": [{"type": "assign", "target": "team1"}],
                    "condition_logic": "OR",
                    "priority": 1,
                    "enabled": False,
                    "description": "A rule",
                    "inbox_id": "inbox_1",
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["condition_logic"] == "OR"
            assert call_kwargs["priority"] == 1
            assert call_kwargs["enabled"] is False
            assert call_kwargs["description"] == "A rule"
            assert call_kwargs["inbox_id"] == "inbox_1"
            assert call_kwargs["created_by"] == "user-42"

    @pytest.mark.asyncio
    async def test_handler_returns_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Rate limit exceeded"}
        with patch(
            f"{_HANDLER_MOD}.handle_create_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_routing_rule(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
                    "actions": [{"type": "assign", "target": "team1"}],
                }
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_default_condition_logic(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rule": {"id": "rule_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_routing_rule(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
                    "actions": [{"type": "assign", "target": "team1"}],
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["condition_logic"] == "AND"

    @pytest.mark.asyncio
    async def test_default_priority(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rule": {"id": "rule_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_routing_rule(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
                    "actions": [{"type": "assign", "target": "team1"}],
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["priority"] == 5

    @pytest.mark.asyncio
    async def test_default_enabled(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rule": {"id": "rule_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_routing_rule(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "conditions": [{"field": "subject", "operator": "contains", "value": "x"}],
                    "actions": [{"type": "assign", "target": "team1"}],
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["enabled"] is True


# ===========================================================================
# handle_get_routing_rules
# ===========================================================================


class TestGetRoutingRules:
    """Tests for GET /api/v1/inbox/routing/rules."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {
            "success": True,
            "rules": [{"id": "rule_1"}],
            "total": 1,
            "limit": 100,
            "offset": 0,
        }
        with patch(
            f"{_HANDLER_MOD}.handle_list_routing_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_routing_rules({"workspace_id": "ws_1"})
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler: SharedInboxHandler):
        result = await handler.handle_get_routing_rules({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_with_filters(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rules": [], "total": 0, "limit": 10, "offset": 5}
        with patch(
            f"{_HANDLER_MOD}.handle_list_routing_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_routing_rules(
                {
                    "workspace_id": "ws_1",
                    "enabled_only": "true",
                    "limit": "10",
                    "offset": "5",
                    "inbox_id": "inbox_1",
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["enabled_only"] is True
            assert call_kwargs["limit"] == 10
            assert call_kwargs["offset"] == 5
            assert call_kwargs["inbox_id"] == "inbox_1"

    @pytest.mark.asyncio
    async def test_enabled_only_false(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rules": [], "total": 0, "limit": 100, "offset": 0}
        with patch(
            f"{_HANDLER_MOD}.handle_list_routing_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_routing_rules(
                {
                    "workspace_id": "ws_1",
                    "enabled_only": "false",
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["enabled_only"] is False

    @pytest.mark.asyncio
    async def test_enabled_only_default(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rules": [], "total": 0, "limit": 100, "offset": 0}
        with patch(
            f"{_HANDLER_MOD}.handle_list_routing_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_routing_rules({"workspace_id": "ws_1"})
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["enabled_only"] is False

    @pytest.mark.asyncio
    async def test_default_limit_and_offset(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rules": [], "total": 0, "limit": 100, "offset": 0}
        with patch(
            f"{_HANDLER_MOD}.handle_list_routing_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_routing_rules({"workspace_id": "ws_1"})
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["limit"] == 100
            assert call_kwargs["offset"] == 0

    @pytest.mark.asyncio
    async def test_handler_returns_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Access denied"}
        with patch(
            f"{_HANDLER_MOD}.handle_list_routing_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_routing_rules({"workspace_id": "ws_1"})
        assert _status(result) == 400


# ===========================================================================
# handle_patch_routing_rule
# ===========================================================================


class TestPatchRoutingRule:
    """Tests for PATCH /api/v1/inbox/routing/rules/:id."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rule": {"id": "rule_1", "enabled": False}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_patch_routing_rule({"enabled": False}, "rule_1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_not_found(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Rule not found"}
        with patch(
            f"{_HANDLER_MOD}.handle_update_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_patch_routing_rule({"enabled": False}, "nonexistent")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_passes_data_and_rule_id(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rule": {"id": "rule_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_update_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_patch_routing_rule({"priority": 1, "name": "New Name"}, "rule_1")
            args = mock_fn.call_args[0]
            assert args[0] == "rule_1"
            assert args[1] == {"priority": 1, "name": "New Name"}


# ===========================================================================
# handle_delete_routing_rule
# ===========================================================================


class TestDeleteRoutingRule:
    """Tests for DELETE /api/v1/inbox/routing/rules/:id."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "deleted": "rule_1"}
        with patch(
            f"{_HANDLER_MOD}.handle_delete_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_delete_routing_rule("rule_1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_not_found(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Rule not found"}
        with patch(
            f"{_HANDLER_MOD}.handle_delete_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_delete_routing_rule("nonexistent")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_passes_rule_id(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "deleted": "rule_abc"}
        with patch(
            f"{_HANDLER_MOD}.handle_delete_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_delete_routing_rule("rule_abc")
            mock_fn.assert_called_once_with("rule_abc")


# ===========================================================================
# handle_post_test_routing_rule
# ===========================================================================


class TestPostTestRoutingRule:
    """Tests for POST /api/v1/inbox/routing/rules/:id/test."""

    @pytest.mark.asyncio
    async def test_success(self, handler: SharedInboxHandler):
        mock_result = {
            "success": True,
            "rule_id": "rule_1",
            "match_count": 5,
            "rule": {"id": "rule_1", "name": "Test Rule"},
        }
        with patch(
            f"{_HANDLER_MOD}.handle_test_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_test_routing_rule({"workspace_id": "ws_1"}, "rule_1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler: SharedInboxHandler):
        result = await handler.handle_post_test_routing_rule({}, "rule_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handler_returns_error(self, handler: SharedInboxHandler):
        mock_result = {"success": False, "error": "Rule not found"}
        with patch(
            f"{_HANDLER_MOD}.handle_test_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_test_routing_rule({"workspace_id": "ws_1"}, "rule_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_passes_rule_id_and_workspace(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "rule_id": "rule_1", "match_count": 0, "rule": {}}
        with patch(
            f"{_HANDLER_MOD}.handle_test_routing_rule",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_test_routing_rule({"workspace_id": "ws_1"}, "rule_1")
            mock_fn.assert_called_once_with("rule_1", "ws_1")


# ===========================================================================
# ROUTES and ROUTE_PREFIXES class attributes
# ===========================================================================


class TestRouteAttributes:
    """Tests for ROUTES and ROUTE_PREFIXES class-level attributes."""

    def test_routes_includes_shared(self, handler: SharedInboxHandler):
        assert "/api/v1/inbox/shared" in handler.ROUTES

    def test_routes_includes_routing_rules(self, handler: SharedInboxHandler):
        assert "/api/v1/inbox/routing/rules" in handler.ROUTES

    def test_route_prefixes_includes_shared(self, handler: SharedInboxHandler):
        assert "/api/v1/inbox/shared/" in handler.ROUTE_PREFIXES

    def test_route_prefixes_includes_routing_rules(self, handler: SharedInboxHandler):
        assert "/api/v1/inbox/routing/rules/" in handler.ROUTE_PREFIXES


# ===========================================================================
# Edge cases and integration-style tests
# ===========================================================================


class TestEdgeCases:
    """Edge-case and boundary tests."""

    @pytest.mark.asyncio
    async def test_post_inbox_null_description(self, handler: SharedInboxHandler):
        """Null description should not be sanitized."""
        mock_result = {"success": True, "inbox": {"id": "inbox_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "description": None,
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["description"] is None

    @pytest.mark.asyncio
    async def test_post_inbox_empty_description(self, handler: SharedInboxHandler):
        """Empty string description should not be sanitized."""
        mock_result = {"success": True, "inbox": {"id": "inbox_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "Test",
                    "description": "",
                }
            )
            call_kwargs = mock_fn.call_args[1]
            # Empty string is falsy, so description should be None
            assert call_kwargs["description"] is None

    @pytest.mark.asyncio
    async def test_assign_message_sanitizes_user_id(self, handler: SharedInboxHandler):
        """The assigned_to field should be sanitized before passing to handler."""
        mock_result = {"success": True, "message": {"id": "msg_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_assign_message",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_assign_message(
                {"assigned_to": "user-valid"}, "inbox_1", "msg_1"
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["assigned_to"] is not None

    @pytest.mark.asyncio
    async def test_status_update_all_valid_statuses(self, handler: SharedInboxHandler):
        """All MessageStatus values should be accepted."""
        from aragora.server.handlers.shared_inbox.models import MessageStatus

        for status_val in MessageStatus:
            mock_result = {"success": True, "message": {"id": "msg_1", "status": status_val.value}}
            with patch(
                f"{_HANDLER_MOD}.handle_update_message_status",
                new_callable=AsyncMock,
                return_value=mock_result,
            ):
                result = await handler.handle_post_update_status(
                    {"status": status_val.value}, "inbox_1", "msg_1"
                )
            assert _status(result) == 200, f"Status {status_val.value} should be valid"

    @pytest.mark.asyncio
    async def test_routing_rule_missing_all_required(self, handler: SharedInboxHandler):
        """Providing no fields at all should return 400."""
        result = await handler.handle_post_routing_rule({})
        assert _status(result) == 400

    def test_handler_init_stores_context(self):
        """Handler should store the context dict."""
        ctx = {"key": "value"}
        h = SharedInboxHandler(ctx)
        assert h.ctx["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_routing_rules_enabled_only_case_insensitive(
        self, handler: SharedInboxHandler
    ):
        """enabled_only 'True' (uppercase T) should still be truthy."""
        mock_result = {"success": True, "rules": [], "total": 0, "limit": 100, "offset": 0}
        with patch(
            f"{_HANDLER_MOD}.handle_list_routing_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_routing_rules(
                {
                    "workspace_id": "ws_1",
                    "enabled_only": "True",
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["enabled_only"] is True

    @pytest.mark.asyncio
    async def test_get_routing_rules_enabled_only_TRUE(self, handler: SharedInboxHandler):
        """enabled_only 'TRUE' (all caps) should also be truthy."""
        mock_result = {"success": True, "rules": [], "total": 0, "limit": 100, "offset": 0}
        with patch(
            f"{_HANDLER_MOD}.handle_list_routing_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_routing_rules(
                {
                    "workspace_id": "ws_1",
                    "enabled_only": "TRUE",
                }
            )
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["enabled_only"] is True

    @pytest.mark.asyncio
    async def test_tag_with_numbers(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "message": {"id": "msg_1", "tags": ["tag123"]}}
        with patch(
            f"{_HANDLER_MOD}.handle_add_message_tag",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_add_tag({"tag": "tag123"}, "inbox_1", "msg_1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_tag_special_chars_rejected(self, handler: SharedInboxHandler):
        """Tags with @, #, etc. should be rejected."""
        result = await handler.handle_post_add_tag({"tag": "@mention"}, "inbox_1", "msg_1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_inbox_with_valid_email(self, handler: SharedInboxHandler):
        mock_result = {"success": True, "inbox": {"id": "inbox_1"}}
        with patch(
            f"{_HANDLER_MOD}.handle_create_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws_1",
                    "name": "Support",
                    "email_address": "support@company.com",
                }
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_create_inbox_email_missing_domain_dot(self, handler: SharedInboxHandler):
        result = await handler.handle_post_shared_inbox(
            {
                "workspace_id": "ws_1",
                "name": "Test",
                "email_address": "user@nodot",
            }
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_inbox_unknown_error_fallback(self, handler: SharedInboxHandler):
        """When handler returns success=False with no error field."""
        mock_result = {"success": False}
        with patch(
            f"{_HANDLER_MOD}.handle_get_shared_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_shared_inbox({}, "inbox_1")
        assert _status(result) == 404
        body = _body(result)
        assert "Unknown error" in body.get("error", "")
