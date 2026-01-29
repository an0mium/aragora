"""Tests for Gmail Threads Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.gmail_threads import (
    GmailThreadsHandler,
    GMAIL_READ_PERMISSION,
    GMAIL_WRITE_PERMISSION,
)


@pytest.fixture
def handler():
    """Create handler instance."""
    return GmailThreadsHandler({})


class TestGmailThreadsHandler:
    """Tests for GmailThreadsHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(GmailThreadsHandler, "ROUTES")
        routes = GmailThreadsHandler.ROUTES
        assert "/api/v1/gmail/threads" in routes
        assert "/api/v1/gmail/drafts" in routes

    def test_handler_route_prefixes(self):
        """Test that handler has route prefixes."""
        assert hasattr(GmailThreadsHandler, "ROUTE_PREFIXES")
        prefixes = GmailThreadsHandler.ROUTE_PREFIXES
        assert "/api/v1/gmail/threads/" in prefixes
        assert "/api/v1/gmail/drafts/" in prefixes

    def test_can_handle_threads_routes(self, handler):
        """Test can_handle for thread routes."""
        assert handler.can_handle("/api/v1/gmail/threads") is True
        assert handler.can_handle("/api/v1/gmail/threads/thread123") is True
        assert handler.can_handle("/api/v1/gmail/threads/thread123/archive") is True

    def test_can_handle_drafts_routes(self, handler):
        """Test can_handle for draft routes."""
        assert handler.can_handle("/api/v1/gmail/drafts") is True
        assert handler.can_handle("/api/v1/gmail/drafts/draft123") is True
        assert handler.can_handle("/api/v1/gmail/drafts/draft123/send") is True

    def test_can_handle_attachment_routes(self, handler):
        """Test can_handle for attachment routes."""
        assert handler.can_handle("/api/v1/gmail/messages/msg123/attachments/att456") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/outlook/threads") is False
        assert handler.can_handle("/api/v1/invalid/route") is False


class TestGmailThreadsAuthentication:
    """Tests for Gmail threads authentication."""

    @pytest.mark.asyncio
    async def test_handle_requires_authentication(self):
        """Test handle method requires authentication."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle("/api/v1/gmail/threads", {}, mock_handler)
            assert result is not None
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_handle_post_requires_authentication(self):
        """Test handle_post method requires authentication."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle_post("/api/v1/gmail/drafts", {}, mock_handler)
            assert result is not None
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_handle_delete_requires_authentication(self):
        """Test handle_delete method requires authentication."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle_delete("/api/v1/gmail/drafts/draft123", {}, mock_handler)
            assert result is not None
            assert result.status_code == 401


class TestGmailThreadOperations:
    """Tests for Gmail thread operations."""

    @pytest.mark.asyncio
    async def test_list_threads_not_connected(self):
        """Test list threads fails when not connected."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle("/api/v1/gmail/threads", {}, mock_handler)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_get_thread_not_connected(self):
        """Test get thread fails when not connected."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle("/api/v1/gmail/threads/thread123", {}, mock_handler)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_archive_thread_not_connected(self):
        """Test archive thread fails when not connected."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle_post(
                "/api/v1/gmail/threads/thread123/archive", {}, mock_handler
            )
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_modify_labels_requires_labels(self):
        """Test modify labels requires add or remove labels."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        mock_state = MagicMock()
        mock_state.refresh_token = "test_token"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = mock_state

            result = await handler.handle_post(
                "/api/v1/gmail/threads/thread123/labels", {}, mock_handler
            )
            assert result.status_code == 400


class TestGmailDraftOperations:
    """Tests for Gmail draft operations."""

    @pytest.mark.asyncio
    async def test_list_drafts_not_connected(self):
        """Test list drafts fails when not connected."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle("/api/v1/gmail/drafts", {}, mock_handler)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_get_draft_not_connected(self):
        """Test get draft fails when not connected."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle("/api/v1/gmail/drafts/draft123", {}, mock_handler)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_delete_draft_not_connected(self):
        """Test delete draft fails when not connected."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle_delete("/api/v1/gmail/drafts/draft123", {}, mock_handler)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_update_draft_not_connected(self):
        """Test update draft fails when not connected."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle_put("/api/v1/gmail/drafts/draft123", {}, mock_handler)
            assert result.status_code == 401


class TestGmailAttachments:
    """Tests for Gmail attachment operations."""

    @pytest.mark.asyncio
    async def test_get_attachment_not_connected(self):
        """Test get attachment fails when not connected."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.gmail_threads.get_user_state"
            ) as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle(
                "/api/v1/gmail/messages/msg123/attachments/att456", {}, mock_handler
            )
            assert result.status_code == 401


class TestGmailPermissions:
    """Tests for Gmail permission checks."""

    @pytest.mark.asyncio
    async def test_read_operations_require_read_permission(self):
        """Test read operations check gmail:read permission."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = MagicMock()

            from aragora.server.handlers.secure import ForbiddenError

            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle("/api/v1/gmail/threads", {}, mock_handler)
            assert result.status_code == 403
            mock_check.assert_called_with(mock_auth.return_value, GMAIL_READ_PERMISSION)

    @pytest.mark.asyncio
    async def test_write_operations_require_write_permission(self):
        """Test write operations check gmail:write permission."""
        handler = GmailThreadsHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = MagicMock()

            from aragora.server.handlers.secure import ForbiddenError

            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle_post("/api/v1/gmail/drafts", {}, mock_handler)
            assert result.status_code == 403
            mock_check.assert_called_with(mock_auth.return_value, GMAIL_WRITE_PERMISSION)
