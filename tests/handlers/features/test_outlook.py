"""Tests for the Outlook/Microsoft 365 email handler.

Tests the Outlook API endpoints including:
- GET  /api/v1/outlook/oauth/url          - Get OAuth authorization URL
- POST /api/v1/outlook/oauth/callback     - Handle OAuth callback
- GET  /api/v1/outlook/folders            - List mail folders
- GET  /api/v1/outlook/messages           - List messages
- GET  /api/v1/outlook/messages/{id}      - Get message details
- GET  /api/v1/outlook/conversations/{id} - Get conversation thread
- POST /api/v1/outlook/send               - Send new message
- POST /api/v1/outlook/reply              - Reply to message
- GET  /api/v1/outlook/search             - Search messages
- POST /api/v1/outlook/messages/{id}/read - Mark as read/unread
- POST /api/v1/outlook/messages/{id}/move - Move message
- DELETE /api/v1/outlook/messages/{id}    - Delete message
- GET  /api/v1/outlook/status             - Connection status
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.outlook import (
    OutlookHandler,
    _get_connector_key,
    _outlook_connectors,
    _oauth_states,
    _storage_lock,
    handle_delete_message,
    handle_get_conversation,
    handle_get_message,
    handle_get_oauth_url,
    handle_get_status,
    handle_list_folders,
    handle_list_messages,
    handle_mark_read,
    handle_move_message,
    handle_oauth_callback,
    handle_reply_message,
    handle_search_messages,
    handle_send_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract decoded JSON body from HandlerResult."""
    if isinstance(result.body, bytes):
        return json.loads(result.body.decode("utf-8"))
    return json.loads(result.body)


def _data(result) -> dict[str, Any]:
    """Extract 'data' from a success_response envelope."""
    return _body(result).get("data", {})


def _error(result) -> str:
    """Extract error message from an error_response envelope."""
    return _body(result).get("error", "")


# ---------------------------------------------------------------------------
# Mock Objects
# ---------------------------------------------------------------------------


@dataclass
class MockFolder:
    id: str = "folder-1"
    display_name: str = "Inbox"
    unread_item_count: int = 5
    total_item_count: int = 100
    child_folder_count: int = 2
    is_hidden: bool = False


@dataclass
class MockMessage:
    id: str = "msg-1"
    thread_id: str = "thread-1"
    subject: str = "Test Subject"
    from_address: str = "sender@example.com"
    to_addresses: list[str] = field(default_factory=lambda: ["recipient@example.com"])
    cc_addresses: list[str] = field(default_factory=list)
    bcc_addresses: list[str] = field(default_factory=list)
    date: datetime = field(default_factory=lambda: datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc))
    body_text: str = "Hello, World!"
    body_html: str = "<p>Hello, World!</p>"
    snippet: str = "Hello, World!"
    is_read: bool = False
    is_starred: bool = False
    is_important: bool = False
    labels: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "from_address": self.from_address,
            "to_addresses": self.to_addresses,
            "date": self.date.isoformat() if self.date else None,
            "snippet": self.snippet,
            "is_read": self.is_read,
            "is_starred": self.is_starred,
            "is_important": self.is_important,
        }


@dataclass
class MockThread:
    id: str = "thread-1"
    subject: str = "Test Thread"
    message_count: int = 3
    participants: list[str] = field(default_factory=lambda: ["a@example.com", "b@example.com"])
    last_message_date: datetime = field(
        default_factory=lambda: datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    )
    snippet: str = "Latest message snippet"
    messages: list[MockMessage] = field(default_factory=list)


@dataclass
class MockSearchResult:
    source_id: str = "msg-search-1"
    id: str = "msg-search-1"
    title: str = "Search Result"
    content: str = "This is the content of the search result message"
    author: str = "author@example.com"
    url: str = "https://outlook.office.com/mail/id/msg-search-1"


@dataclass
class MockAttachment:
    id: str = "attach-1"
    filename: str = "report.pdf"
    mime_type: str = "application/pdf"
    size: int = 1024


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_outlook_state():
    """Clear global state before/after each test."""
    with _storage_lock:
        _outlook_connectors.clear()
        _oauth_states.clear()
    yield
    with _storage_lock:
        _outlook_connectors.clear()
        _oauth_states.clear()


@pytest.fixture
def handler():
    """Create an OutlookHandler instance."""
    return OutlookHandler(server_context={})


@pytest.fixture
def mock_connector():
    """Create a mock OutlookConnector."""
    connector = MagicMock()
    connector.is_configured.return_value = True
    connector.get_oauth_url.return_value = "https://login.microsoftonline.com/auth?state=xyz"
    connector.authenticate = AsyncMock(return_value=True)
    connector.get_user_info = AsyncMock(
        return_value={
            "mail": "user@example.com",
            "userPrincipalName": "user@example.com",
            "displayName": "Test User",
        }
    )
    connector.list_folders = AsyncMock(return_value=[MockFolder()])
    connector.list_messages = AsyncMock(return_value=(["msg-1", "msg-2"], "next-page-token"))
    connector.get_message = AsyncMock(return_value=MockMessage())
    connector.get_message_attachments = AsyncMock(return_value=[MockAttachment()])
    connector.get_conversation = AsyncMock(
        return_value=MockThread(messages=[MockMessage(id="m1"), MockMessage(id="m2")])
    )
    connector.send_message = AsyncMock(return_value={"success": True})
    connector.reply_to_message = AsyncMock(return_value={"success": True})
    connector.search = AsyncMock(return_value=[MockSearchResult()])
    connector._api_request = AsyncMock(return_value={"success": True})
    return connector


@pytest.fixture
def stored_connector(mock_connector):
    """Store a mock connector for a default workspace/user so auth-required endpoints work."""
    key = _get_connector_key("default", "default")
    with _storage_lock:
        _outlook_connectors[key] = mock_connector
    return mock_connector


# =============================================================================
# Handler Initialization Tests
# =============================================================================


class TestOutlookHandlerInit:
    """Tests for handler initialization."""

    def test_create_with_server_context(self):
        h = OutlookHandler(server_context={})
        assert h.ctx == {}

    def test_create_with_ctx(self):
        h = OutlookHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/v1/outlook/oauth/url" in handler.ROUTES
        assert "/api/v1/outlook/send" in handler.ROUTES

    def test_route_prefixes_defined(self, handler):
        assert len(handler.ROUTE_PREFIXES) > 0
        assert "/api/v1/outlook/messages/" in handler.ROUTE_PREFIXES
        assert "/api/v1/outlook/conversations/" in handler.ROUTE_PREFIXES

    def test_can_handle_known_routes(self, handler):
        assert handler.can_handle("/api/v1/outlook/oauth/url") is True
        assert handler.can_handle("/api/v1/outlook/folders") is True
        assert handler.can_handle("/api/v1/outlook/messages") is True
        assert handler.can_handle("/api/v1/outlook/send") is True
        assert handler.can_handle("/api/v1/outlook/reply") is True
        assert handler.can_handle("/api/v1/outlook/search") is True
        assert handler.can_handle("/api/v1/outlook/status") is True

    def test_can_handle_prefix_routes(self, handler):
        assert handler.can_handle("/api/v1/outlook/messages/msg-123") is True
        assert handler.can_handle("/api/v1/outlook/conversations/conv-456") is True
        assert handler.can_handle("/api/v1/outlook/messages/msg-1/read") is True
        assert handler.can_handle("/api/v1/outlook/messages/msg-1/move") is True

    def test_cannot_handle_unknown(self, handler):
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/outlook") is False

    def test_handle_returns_none(self, handler):
        result = handler.handle("/api/v1/outlook/folders", {}, MagicMock())
        assert result is None


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetConnectorKey:
    def test_basic_key(self):
        assert _get_connector_key("ws1", "user1") == "ws1:user1"

    def test_different_users(self):
        k1 = _get_connector_key("ws1", "userA")
        k2 = _get_connector_key("ws1", "userB")
        assert k1 != k2

    def test_different_workspaces(self):
        k1 = _get_connector_key("ws1", "user1")
        k2 = _get_connector_key("ws2", "user1")
        assert k1 != k2


class TestGetUserID:
    def test_default_user(self, handler):
        assert handler._get_user_id() == "default"

    def test_user_from_auth_context(self):
        auth = MagicMock()
        auth.user_id = "usr-42"
        h = OutlookHandler(server_context={"auth_context": auth})
        assert h._get_user_id() == "usr-42"

    def test_user_without_user_id_attr(self):
        auth = object()
        h = OutlookHandler(server_context={"auth_context": auth})
        assert h._get_user_id() == "default"


# =============================================================================
# _get_or_create_connector Tests
# =============================================================================


class TestGetOrCreateConnector:
    @pytest.mark.asyncio
    async def test_returns_cached_connector(self, mock_connector):
        key = _get_connector_key("ws1", "u1")
        with _storage_lock:
            _outlook_connectors[key] = mock_connector
        from aragora.server.handlers.features.outlook import _get_or_create_connector

        result = await _get_or_create_connector("ws1", "u1")
        assert result is mock_connector

    @pytest.mark.asyncio
    async def test_returns_none_when_import_fails(self):
        from aragora.server.handlers.features.outlook import _get_or_create_connector

        # Patch the import inside _get_or_create_connector to raise ImportError
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "outlook" in name and "communication" in name:
                raise ImportError("OutlookConnector not available")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = await _get_or_create_connector("import-fail-ws", "import-fail-user")
        assert result is None


# =============================================================================
# OAuth URL Tests (standalone function)
# =============================================================================


class TestHandleGetOAuthUrl:
    @pytest.mark.asyncio
    async def test_success(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_oauth_url.__wrapped__(
                workspace_id="default",
                user_id="user1",
                redirect_uri="http://localhost/callback",
            )
        assert result["success"] is True
        assert "auth_url" in result
        assert "state" in result

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_get_oauth_url.__wrapped__(
                workspace_id="default",
                user_id="user1",
                redirect_uri="http://localhost/callback",
            )
        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_connector_not_configured(self, mock_connector):
        mock_connector.is_configured.return_value = False
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_oauth_url.__wrapped__(
                workspace_id="default",
                user_id="user1",
                redirect_uri="http://localhost/callback",
            )
        assert result["success"] is False
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_state_stored_in_oauth_states(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_oauth_url.__wrapped__(
                workspace_id="ws1",
                user_id="user1",
                redirect_uri="http://localhost/cb",
            )
        state = result["state"]
        assert state in _oauth_states
        assert _oauth_states[state]["workspace_id"] == "ws1"
        assert _oauth_states[state]["user_id"] == "user1"
        assert _oauth_states[state]["redirect_uri"] == "http://localhost/cb"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.get_oauth_url.side_effect = ConnectionError("network error")
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_oauth_url.__wrapped__(
                workspace_id="default",
                user_id="user1",
                redirect_uri="http://localhost/callback",
            )
        assert result["success"] is False
        assert "Failed to generate OAuth URL" in result["error"]


# =============================================================================
# OAuth Callback Tests
# =============================================================================


class TestHandleOAuthCallback:
    @pytest.mark.asyncio
    async def test_success(self, mock_connector):
        # Pre-store state
        _oauth_states["test-state"] = {
            "workspace_id": "ws1",
            "user_id": "u1",
            "redirect_uri": "http://localhost/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_oauth_callback(
                code="auth-code-123",
                state="test-state",
            )
        assert result["success"] is True
        assert result["email"] == "user@example.com"
        assert result["workspace_id"] == "ws1"
        assert result["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_invalid_state(self):
        result = await handle_oauth_callback(
            code="auth-code-123",
            state="invalid-state",
        )
        assert result["success"] is False
        assert "Invalid or expired state" in result["error"]

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        _oauth_states["state-1"] = {
            "workspace_id": "ws1",
            "user_id": "u1",
            "redirect_uri": "http://localhost/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_oauth_callback(code="code", state="state-1")
        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_auth_fails(self, mock_connector):
        mock_connector.authenticate = AsyncMock(return_value=False)
        _oauth_states["state-2"] = {
            "workspace_id": "ws1",
            "user_id": "u1",
            "redirect_uri": "http://localhost/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_oauth_callback(code="bad-code", state="state-2")
        assert result["success"] is False
        assert "Authentication failed" in result["error"]

    @pytest.mark.asyncio
    async def test_uses_stored_redirect_uri(self, mock_connector):
        _oauth_states["state-3"] = {
            "workspace_id": "ws1",
            "user_id": "u1",
            "redirect_uri": "http://stored-uri/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_oauth_callback(code="code", state="state-3")
        mock_connector.authenticate.assert_awaited_once_with(
            code="code", redirect_uri="http://stored-uri/cb"
        )

    @pytest.mark.asyncio
    async def test_override_redirect_uri(self, mock_connector):
        _oauth_states["state-4"] = {
            "workspace_id": "ws1",
            "user_id": "u1",
            "redirect_uri": "http://stored-uri/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_oauth_callback(
                code="code", state="state-4", redirect_uri="http://override/cb"
            )
        mock_connector.authenticate.assert_awaited_once_with(
            code="code", redirect_uri="http://override/cb"
        )

    @pytest.mark.asyncio
    async def test_user_profile_failure_still_succeeds(self, mock_connector):
        mock_connector.get_user_info = AsyncMock(side_effect=ConnectionError("no profile"))
        _oauth_states["state-5"] = {
            "workspace_id": "ws1",
            "user_id": "u1",
            "redirect_uri": "http://localhost/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_oauth_callback(code="code", state="state-5")
        assert result["success"] is True
        assert result["email"] == ""

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.authenticate = AsyncMock(side_effect=ValueError("bad"))
        _oauth_states["state-6"] = {
            "workspace_id": "ws1",
            "user_id": "u1",
            "redirect_uri": "http://localhost/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_oauth_callback(code="code", state="state-6")
        assert result["success"] is False
        assert "OAuth callback failed" in result["error"]

    @pytest.mark.asyncio
    async def test_state_consumed_on_use(self, mock_connector):
        _oauth_states["state-consume"] = {
            "workspace_id": "ws1",
            "user_id": "u1",
            "redirect_uri": "http://localhost/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            await handle_oauth_callback(code="code", state="state-consume")
        assert "state-consume" not in _oauth_states


# =============================================================================
# List Folders Tests
# =============================================================================


class TestHandleListFolders:
    @pytest.mark.asyncio
    async def test_success(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_folders.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is True
        assert result["total"] == 1
        assert result["folders"][0]["display_name"] == "Inbox"
        assert result["folders"][0]["unread_count"] == 5

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_list_folders.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_multiple_folders(self, mock_connector):
        mock_connector.list_folders = AsyncMock(
            return_value=[
                MockFolder(id="f1", display_name="Inbox"),
                MockFolder(id="f2", display_name="Sent"),
                MockFolder(id="f3", display_name="Drafts"),
            ]
        )
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_folders.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.list_folders = AsyncMock(side_effect=TimeoutError("timeout"))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_folders.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is False
        assert "Failed to list folders" in result["error"]


# =============================================================================
# List Messages Tests
# =============================================================================


class TestHandleListMessages:
    @pytest.mark.asyncio
    async def test_success(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_messages.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is True
        assert result["next_page_token"] == "next-page-token"
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_list_messages.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_with_folder_and_pagination(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                folder_id="folder-inbox",
                max_results=10,
                page_token="page-2",
                filter_query="isRead eq false",
            )
        mock_connector.list_messages.assert_awaited_once_with(
            folder_id="folder-inbox",
            max_results=10,
            page_token="page-2",
            query="isRead eq false",
        )

    @pytest.mark.asyncio
    async def test_message_without_to_dict(self, mock_connector):
        """Test messages that don't have to_dict method use manual dict construction."""
        # Create a plain object without to_dict method
        msg = MagicMock(spec=[])
        msg.id = "msg-no-todict"
        msg.thread_id = "thread-1"
        msg.subject = "Test Subject"
        msg.from_address = "sender@example.com"
        msg.to_addresses = ["r@example.com"]
        msg.date = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
        msg.snippet = "Snippet"
        msg.is_read = False
        msg.is_starred = False
        msg.is_important = False
        mock_connector.get_message = AsyncMock(return_value=msg)
        mock_connector.list_messages = AsyncMock(return_value=(["msg-no-todict"], None))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_messages.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["id"] == "msg-no-todict"

    @pytest.mark.asyncio
    async def test_individual_message_fetch_failure(self, mock_connector):
        """Test that individual message fetch failures are handled gracefully."""
        mock_connector.list_messages = AsyncMock(return_value=(["msg-1", "msg-2"], None))

        call_count = 0

        async def get_message_side_effect(msg_id, include_body=False):
            nonlocal call_count
            call_count += 1
            if msg_id == "msg-1":
                raise ConnectionError("failed")
            return MockMessage(id=msg_id)

        mock_connector.get_message = get_message_side_effect

        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_messages.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is True
        # msg-1 failed, only msg-2 should be in results
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.list_messages = AsyncMock(side_effect=OSError("disk error"))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_messages.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is False
        assert "Failed to list messages" in result["error"]


# =============================================================================
# Get Message Tests
# =============================================================================


class TestHandleGetMessage:
    @pytest.mark.asyncio
    async def test_success(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
            )
        assert result["success"] is True
        msg = result["message"]
        assert msg["id"] == "msg-1"
        assert msg["subject"] == "Test Subject"
        assert msg["from_address"] == "sender@example.com"
        assert msg["body_text"] == "Hello, World!"
        assert "attachments" not in msg

    @pytest.mark.asyncio
    async def test_with_attachments(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
                include_attachments=True,
            )
        assert result["success"] is True
        assert "attachments" in result["message"]
        assert result["message"]["attachments"][0]["filename"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_get_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_message_with_no_date(self, mock_connector):
        mock_connector.get_message = AsyncMock(return_value=MockMessage(date=None))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
            )
        assert result["success"] is True
        assert result["message"]["date"] is None

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.get_message = AsyncMock(side_effect=ValueError("bad"))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
            )
        assert result["success"] is False
        assert "Failed to get message" in result["error"]


# =============================================================================
# Get Conversation Tests
# =============================================================================


class TestHandleGetConversation:
    @pytest.mark.asyncio
    async def test_success(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_conversation.__wrapped__(
                workspace_id="default",
                user_id="user1",
                conversation_id="conv-1",
            )
        assert result["success"] is True
        conv = result["conversation"]
        assert conv["id"] == "thread-1"
        assert conv["subject"] == "Test Thread"
        assert conv["message_count"] == 3
        assert len(conv["messages"]) == 2

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_get_conversation.__wrapped__(
                workspace_id="default",
                user_id="user1",
                conversation_id="conv-1",
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_thread_with_no_last_message_date(self, mock_connector):
        mock_connector.get_conversation = AsyncMock(
            return_value=MockThread(last_message_date=None, messages=[])
        )
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_conversation.__wrapped__(
                workspace_id="default",
                user_id="user1",
                conversation_id="conv-1",
            )
        assert result["success"] is True
        assert result["conversation"]["last_message_date"] is None

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.get_conversation = AsyncMock(side_effect=TimeoutError("slow"))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_conversation.__wrapped__(
                workspace_id="default",
                user_id="user1",
                conversation_id="conv-1",
            )
        assert result["success"] is False
        assert "Failed to get conversation" in result["error"]


# =============================================================================
# Send Message Tests
# =============================================================================


class TestHandleSendMessage:
    @pytest.mark.asyncio
    async def test_success_text(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_send_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                to_addresses=["to@example.com"],
                subject="Test",
                body="Hello",
            )
        assert result["success"] is True
        assert result["message"] == "Email sent successfully"
        mock_connector.send_message.assert_awaited_once_with(
            to=["to@example.com"],
            subject="Test",
            body="Hello",
            cc=None,
            bcc=None,
            html_body=None,
        )

    @pytest.mark.asyncio
    async def test_success_html(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_send_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                to_addresses=["to@example.com"],
                subject="Test",
                body="<p>Hello</p>",
                body_type="html",
            )
        assert result["success"] is True
        mock_connector.send_message.assert_awaited_once_with(
            to=["to@example.com"],
            subject="Test",
            body="",
            cc=None,
            bcc=None,
            html_body="<p>Hello</p>",
        )

    @pytest.mark.asyncio
    async def test_with_cc_and_bcc(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_send_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                to_addresses=["to@example.com"],
                subject="Test",
                body="Hello",
                cc_addresses=["cc@example.com"],
                bcc_addresses=["bcc@example.com"],
            )
        assert result["success"] is True
        mock_connector.send_message.assert_awaited_once_with(
            to=["to@example.com"],
            subject="Test",
            body="Hello",
            cc=["cc@example.com"],
            bcc=["bcc@example.com"],
            html_body=None,
        )

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_send_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                to_addresses=["to@example.com"],
                subject="Test",
                body="Hello",
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.send_message = AsyncMock(side_effect=ConnectionError("fail"))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_send_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                to_addresses=["to@example.com"],
                subject="Test",
                body="Hello",
            )
        assert result["success"] is False
        assert "Failed to send message" in result["error"]


# =============================================================================
# Reply Message Tests
# =============================================================================


class TestHandleReplyMessage:
    @pytest.mark.asyncio
    async def test_success(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_reply_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
                body="Thanks!",
            )
        assert result["success"] is True
        assert result["message"] == "Reply sent successfully"
        assert result["in_reply_to"] == "msg-1"

    @pytest.mark.asyncio
    async def test_reply_all(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_reply_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
                body="Thanks!",
                reply_all=True,
            )
        mock_connector.reply_to_message.assert_awaited_once_with(
            original_message_id="msg-1",
            body="Thanks!",
            cc=None,
            html_body=None,
            reply_all=True,
        )

    @pytest.mark.asyncio
    async def test_reply_html(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_reply_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
                body="<p>Thanks!</p>",
                body_type="html",
            )
        mock_connector.reply_to_message.assert_awaited_once_with(
            original_message_id="msg-1",
            body="<p>Thanks!</p>",
            cc=None,
            html_body="<p>Thanks!</p>",
            reply_all=False,
        )

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_reply_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
                body="Thanks!",
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.reply_to_message = AsyncMock(side_effect=OSError("fail"))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_reply_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
                body="Thanks!",
            )
        assert result["success"] is False
        assert "Failed to reply to message" in result["error"]


# =============================================================================
# Search Messages Tests
# =============================================================================


class TestHandleSearchMessages:
    @pytest.mark.asyncio
    async def test_success(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_search_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                query="test query",
            )
        assert result["success"] is True
        assert result["query"] == "test query"
        assert result["total"] == 1
        assert result["results"][0]["title"] == "Search Result"

    @pytest.mark.asyncio
    async def test_search_result_uses_source_id(self, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_search_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                query="test",
            )
        assert result["results"][0]["id"] == "msg-search-1"

    @pytest.mark.asyncio
    async def test_search_result_without_source_id(self, mock_connector):
        """Test fallback to .id when source_id not present."""
        sr = MagicMock()
        sr.id = "fallback-id"
        sr.title = "Title"
        sr.content = "Content"
        sr.author = "a@b.com"
        sr.url = "http://example.com"
        del sr.source_id  # Remove source_id attribute
        mock_connector.search = AsyncMock(return_value=[sr])
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_search_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                query="test",
            )
        assert result["results"][0]["id"] == "fallback-id"

    @pytest.mark.asyncio
    async def test_search_truncates_content(self, mock_connector):
        sr = MockSearchResult(content="x" * 500)
        mock_connector.search = AsyncMock(return_value=[sr])
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_search_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                query="test",
            )
        assert len(result["results"][0]["snippet"]) == 200

    @pytest.mark.asyncio
    async def test_connector_not_available(self):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_search_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                query="test",
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.search = AsyncMock(side_effect=ValueError("fail"))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_search_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                query="test",
            )
        assert result["success"] is False
        assert "Failed to search messages" in result["error"]


# =============================================================================
# Mark Read Tests
# =============================================================================


class TestHandleMarkRead:
    @pytest.mark.asyncio
    async def test_mark_as_read(self, stored_connector):
        result = await handle_mark_read.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            is_read=True,
        )
        assert result["success"] is True
        assert result["message_id"] == "msg-1"
        assert result["is_read"] is True
        stored_connector._api_request.assert_awaited_once_with(
            "/messages/msg-1", method="PATCH", json_data={"isRead": True}
        )

    @pytest.mark.asyncio
    async def test_mark_as_unread(self, stored_connector):
        result = await handle_mark_read.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            is_read=False,
        )
        assert result["success"] is True
        assert result["is_read"] is False
        stored_connector._api_request.assert_awaited_once_with(
            "/messages/msg-1", method="PATCH", json_data={"isRead": False}
        )

    @pytest.mark.asyncio
    async def test_not_authenticated(self):
        result = await handle_mark_read.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
        )
        assert result["success"] is False
        assert "Not authenticated" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, stored_connector):
        stored_connector._api_request = AsyncMock(side_effect=ConnectionError("fail"))
        result = await handle_mark_read.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
        )
        assert result["success"] is False
        assert "Failed to update message read status" in result["error"]


# =============================================================================
# Move Message Tests
# =============================================================================


class TestHandleMoveMessage:
    @pytest.mark.asyncio
    async def test_success(self, stored_connector):
        result = await handle_move_message.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            destination_folder_id="folder-archive",
        )
        assert result["success"] is True
        assert result["message_id"] == "msg-1"
        assert result["destination_folder_id"] == "folder-archive"
        stored_connector._api_request.assert_awaited_once_with(
            "/messages/msg-1/move",
            method="POST",
            json_data={"destinationId": "folder-archive"},
        )

    @pytest.mark.asyncio
    async def test_not_authenticated(self):
        result = await handle_move_message.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            destination_folder_id="folder-archive",
        )
        assert result["success"] is False
        assert "Not authenticated" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, stored_connector):
        stored_connector._api_request = AsyncMock(side_effect=TimeoutError("fail"))
        result = await handle_move_message.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            destination_folder_id="folder-archive",
        )
        assert result["success"] is False
        assert "Failed to move message" in result["error"]


# =============================================================================
# Delete Message Tests
# =============================================================================


class TestHandleDeleteMessage:
    @pytest.mark.asyncio
    async def test_soft_delete(self, stored_connector):
        deleted_folder = MockFolder(id="deleted-items-id", display_name="Deleted Items")
        stored_connector.list_folders = AsyncMock(
            return_value=[MockFolder(), deleted_folder]
        )
        result = await handle_delete_message.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            permanent=False,
        )
        assert result["success"] is True
        assert result["deleted"] is True
        assert result["permanent"] is False
        stored_connector._api_request.assert_awaited_once_with(
            "/messages/msg-1/move",
            method="POST",
            json_data={"destinationId": "deleted-items-id"},
        )

    @pytest.mark.asyncio
    async def test_soft_delete_no_deleted_folder_fallback(self, stored_connector):
        """When no 'Deleted Items' folder, fallback to permanent delete."""
        stored_connector.list_folders = AsyncMock(
            return_value=[MockFolder(display_name="Inbox")]
        )
        result = await handle_delete_message.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            permanent=False,
        )
        assert result["success"] is True
        stored_connector._api_request.assert_awaited_once_with(
            "/messages/msg-1", method="DELETE"
        )

    @pytest.mark.asyncio
    async def test_permanent_delete(self, stored_connector):
        result = await handle_delete_message.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            permanent=True,
        )
        assert result["success"] is True
        assert result["permanent"] is True
        stored_connector._api_request.assert_awaited_once_with(
            "/messages/msg-1", method="DELETE"
        )

    @pytest.mark.asyncio
    async def test_not_authenticated(self):
        result = await handle_delete_message.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
        )
        assert result["success"] is False
        assert "Not authenticated" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, stored_connector):
        stored_connector._api_request = AsyncMock(side_effect=OSError("fail"))
        result = await handle_delete_message.__wrapped__(
            workspace_id="default",
            user_id="default",
            message_id="msg-1",
            permanent=True,
        )
        assert result["success"] is False
        assert "Failed to delete message" in result["error"]


# =============================================================================
# Get Status Tests
# =============================================================================


class TestHandleGetStatus:
    @pytest.mark.asyncio
    async def test_not_connected(self):
        result = await handle_get_status.__wrapped__(
            workspace_id="default", user_id="default"
        )
        assert result["success"] is True
        assert result["connected"] is False
        assert result["email"] is None

    @pytest.mark.asyncio
    async def test_connected(self, stored_connector):
        result = await handle_get_status.__wrapped__(
            workspace_id="default", user_id="default"
        )
        assert result["success"] is True
        assert result["connected"] is True
        assert result["email"] == "user@example.com"
        assert result["display_name"] == "Test User"

    @pytest.mark.asyncio
    async def test_token_expired(self, stored_connector):
        stored_connector.get_user_info = AsyncMock(
            side_effect=PermissionError("token expired")
        )
        result = await handle_get_status.__wrapped__(
            workspace_id="default", user_id="default"
        )
        assert result["success"] is True
        assert result["connected"] is False
        assert "Token expired" in result["error"]

    @pytest.mark.asyncio
    async def test_profile_uses_upn_fallback(self, stored_connector):
        stored_connector.get_user_info = AsyncMock(
            return_value={"userPrincipalName": "upn@example.com", "displayName": "UPN User"}
        )
        result = await handle_get_status.__wrapped__(
            workspace_id="default", user_id="default"
        )
        assert result["success"] is True
        assert result["email"] == "upn@example.com"

    @pytest.mark.asyncio
    async def test_outer_exception_returns_error(self, stored_connector):
        """Test that a KeyError in the outer try block is caught."""
        stored_connector.get_user_info = AsyncMock(side_effect=KeyError("missing_key"))
        result = await handle_get_status.__wrapped__(
            workspace_id="default", user_id="default"
        )
        assert result["success"] is False
        assert "Failed to get connection status" in result["error"]


# =============================================================================
# Handler Class Method Tests
# =============================================================================


class TestOutlookHandlerMethods:
    """Test the class-level handler methods that wrap the standalone functions."""

    @pytest.mark.asyncio
    async def test_handle_get_oauth_url_missing_redirect_uri(self, handler):
        result = await handler.handle_get_oauth_url({})
        assert _status(result) == 400
        assert "redirect_uri required" in _error(result)

    @pytest.mark.asyncio
    async def test_handle_get_oauth_url_success(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_get_oauth_url(
                {"redirect_uri": "http://localhost/cb", "workspace_id": "ws1"}
            )
        assert _status(result) == 200
        data = _data(result)
        assert data["success"] is True
        assert "auth_url" in data

    @pytest.mark.asyncio
    async def test_handle_get_oauth_url_error(self, handler):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_get_oauth_url(
                {"redirect_uri": "http://localhost/cb"}
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handle_post_oauth_callback_missing_fields(self, handler):
        result = await handler.handle_post_oauth_callback({})
        assert _status(result) == 400
        assert "code and state required" in _error(result)

    @pytest.mark.asyncio
    async def test_handle_post_oauth_callback_missing_state(self, handler):
        result = await handler.handle_post_oauth_callback({"code": "abc"})
        assert _status(result) == 400
        assert "code and state required" in _error(result)

    @pytest.mark.asyncio
    async def test_handle_post_oauth_callback_success(self, handler, mock_connector):
        _oauth_states["test-state"] = {
            "workspace_id": "default",
            "user_id": "default",
            "redirect_uri": "http://localhost/cb",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_post_oauth_callback(
                {"code": "auth-code", "state": "test-state"}
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_get_folders_success(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_get_folders({})
        assert _status(result) == 200
        data = _data(result)
        assert data["success"] is True
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_handle_get_folders_error(self, handler):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_get_folders({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handle_get_messages_success(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_get_messages({})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_get_messages_with_params(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_get_messages({
                "folder_id": "inbox",
                "max_results": "10",
                "page_token": "page2",
                "filter": "isRead eq false",
            })
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_get_messages_caps_max_results(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            await handler.handle_get_messages({"max_results": "9999"})
        # Verify max_results was capped to 500
        mock_connector.list_messages.assert_awaited_once()
        call_kwargs = mock_connector.list_messages.call_args[1]
        assert call_kwargs["max_results"] == 500

    @pytest.mark.asyncio
    async def test_handle_get_message_success(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_get_message({}, message_id="msg-1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_get_message_with_attachments(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_get_message(
                {"include_attachments": "true"}, message_id="msg-1"
            )
        assert _status(result) == 200
        data = _data(result)
        assert "attachments" in data["message"]

    @pytest.mark.asyncio
    async def test_handle_get_message_error(self, handler):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_get_message({}, message_id="msg-1")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_handle_get_conversation_success(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_get_conversation({}, conversation_id="conv-1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_get_conversation_caps_max_messages(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            await handler.handle_get_conversation(
                {"max_messages": "9999"}, conversation_id="conv-1"
            )
        mock_connector.get_conversation.assert_awaited_once_with("conv-1", 500)

    @pytest.mark.asyncio
    async def test_handle_get_conversation_error(self, handler):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_get_conversation({}, conversation_id="conv-1")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_handle_post_send_missing_fields(self, handler):
        result = await handler.handle_post_send({})
        assert _status(result) == 400
        assert "to, subject, and body required" in _error(result)

    @pytest.mark.asyncio
    async def test_handle_post_send_missing_subject(self, handler):
        result = await handler.handle_post_send({"to": "a@b.com", "body": "hi"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handle_post_send_missing_body(self, handler):
        result = await handler.handle_post_send({"to": "a@b.com", "subject": "Hi"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handle_post_send_success(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_post_send({
                "to": ["to@example.com"],
                "subject": "Test",
                "body": "Hello",
            })
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_post_send_string_to(self, handler, mock_connector):
        """When 'to' is a single string instead of list, it should be wrapped."""
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            await handler.handle_post_send({
                "to": "single@example.com",
                "subject": "Test",
                "body": "Hello",
            })
        mock_connector.send_message.assert_awaited_once()
        call_kwargs = mock_connector.send_message.call_args[1]
        assert call_kwargs["to"] == ["single@example.com"]

    @pytest.mark.asyncio
    async def test_handle_post_reply_missing_fields(self, handler):
        result = await handler.handle_post_reply({})
        assert _status(result) == 400
        assert "message_id and body required" in _error(result)

    @pytest.mark.asyncio
    async def test_handle_post_reply_missing_body(self, handler):
        result = await handler.handle_post_reply({"message_id": "msg-1"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handle_post_reply_success(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_post_reply({
                "message_id": "msg-1",
                "body": "Thanks!",
            })
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_post_reply_with_options(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_post_reply({
                "message_id": "msg-1",
                "body": "<p>Reply</p>",
                "body_type": "html",
                "reply_all": True,
                "cc": ["cc@example.com"],
            })
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_get_search_missing_query(self, handler):
        result = await handler.handle_get_search({})
        assert _status(result) == 400
        assert "q (query) required" in _error(result)

    @pytest.mark.asyncio
    async def test_handle_get_search_success(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler.handle_get_search({"q": "test"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_get_search_caps_max_results(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            await handler.handle_get_search({"q": "test", "max_results": "9999"})
        mock_connector.search.assert_awaited_once_with(query="test", limit=500)

    @pytest.mark.asyncio
    async def test_handle_get_status_success(self, handler, stored_connector):
        result = await handler.handle_get_status({})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_get_status_error(self, handler):
        with patch(
            "aragora.server.handlers.features.outlook.handle_get_status",
            new_callable=AsyncMock,
            return_value={"success": False, "error": "Failed"},
        ):
            result = await handler.handle_get_status({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handle_post_mark_read_success(self, handler, stored_connector):
        result = await handler.handle_post_mark_read(
            {"is_read": True}, message_id="msg-1"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_post_mark_read_error(self, handler):
        result = await handler.handle_post_mark_read({}, message_id="msg-1")
        # Not authenticated
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handle_post_move_missing_destination(self, handler):
        result = await handler.handle_post_move({}, message_id="msg-1")
        assert _status(result) == 400
        assert "destination_folder_id required" in _error(result)

    @pytest.mark.asyncio
    async def test_handle_post_move_success(self, handler, stored_connector):
        result = await handler.handle_post_move(
            {"destination_folder_id": "folder-archive"}, message_id="msg-1"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_delete_message_success(self, handler, stored_connector):
        result = await handler.handle_delete_message({}, message_id="msg-1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_delete_message_permanent(self, handler, stored_connector):
        result = await handler.handle_delete_message(
            {"permanent": "true"}, message_id="msg-1"
        )
        assert _status(result) == 200
        data = _data(result)
        assert data["permanent"] is True

    @pytest.mark.asyncio
    async def test_handle_delete_message_not_authenticated(self, handler):
        result = await handler.handle_delete_message({}, message_id="msg-1")
        assert _status(result) == 400


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    def test_connector_key_special_characters(self):
        key = _get_connector_key("ws:special", "user/slash")
        assert key == "ws:special:user/slash"

    @pytest.mark.asyncio
    async def test_empty_message_list(self, mock_connector):
        mock_connector.list_messages = AsyncMock(return_value=([], None))
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_messages.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is True
        assert result["messages"] == []
        assert result["total"] == 0
        assert result["next_page_token"] is None

    @pytest.mark.asyncio
    async def test_empty_folder_list(self, mock_connector):
        mock_connector.list_folders = AsyncMock(return_value=[])
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_list_folders.__wrapped__(
                workspace_id="default", user_id="user1"
            )
        assert result["success"] is True
        assert result["folders"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_empty_search_results(self, mock_connector):
        mock_connector.search = AsyncMock(return_value=[])
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_search_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                query="nothing matches",
            )
        assert result["success"] is True
        assert result["results"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_search_result_with_empty_content(self, mock_connector):
        sr = MockSearchResult(content="")
        mock_connector.search = AsyncMock(return_value=[sr])
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_search_messages.__wrapped__(
                workspace_id="default",
                user_id="user1",
                query="test",
            )
        assert result["results"][0]["snippet"] == ""

    @pytest.mark.asyncio
    async def test_message_with_all_fields(self, mock_connector):
        full_msg = MockMessage(
            id="full-msg",
            thread_id="full-thread",
            subject="Full Subject",
            from_address="from@example.com",
            to_addresses=["to1@example.com", "to2@example.com"],
            cc_addresses=["cc@example.com"],
            bcc_addresses=["bcc@example.com"],
            date=datetime(2025, 6, 15, 14, 30, tzinfo=timezone.utc),
            body_text="Plain text",
            body_html="<p>HTML</p>",
            snippet="Snippet text",
            is_read=True,
            is_starred=True,
            is_important=True,
            labels=["important", "starred"],
        )
        mock_connector.get_message = AsyncMock(return_value=full_msg)
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="full-msg",
            )
        msg = result["message"]
        assert msg["cc_addresses"] == ["cc@example.com"]
        assert msg["bcc_addresses"] == ["bcc@example.com"]
        assert msg["is_read"] is True
        assert msg["is_starred"] is True
        assert msg["is_important"] is True
        assert msg["labels"] == ["important", "starred"]
        assert msg["body_html"] == "<p>HTML</p>"

    @pytest.mark.asyncio
    async def test_handler_default_workspace(self, handler, mock_connector):
        """Test that handler uses 'default' workspace when not specified."""
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            await handler.handle_get_folders({})
        mock_connector.list_folders.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_conversation_with_message_dates(self, mock_connector):
        thread = MockThread(
            messages=[
                MockMessage(id="m1", date=datetime(2025, 1, 1, tzinfo=timezone.utc)),
                MockMessage(id="m2", date=None),
            ]
        )
        mock_connector.get_conversation = AsyncMock(return_value=thread)
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_get_conversation.__wrapped__(
                workspace_id="default",
                user_id="user1",
                conversation_id="conv-1",
            )
        msgs = result["conversation"]["messages"]
        assert msgs[0]["date"] == "2025-01-01T00:00:00+00:00"
        assert msgs[1]["date"] is None

    @pytest.mark.asyncio
    async def test_send_returns_false_success(self, mock_connector):
        """Test when send_message returns success=False."""
        mock_connector.send_message = AsyncMock(return_value={"success": False})
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_send_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                to_addresses=["to@example.com"],
                subject="Test",
                body="Hello",
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_reply_returns_false_success(self, mock_connector):
        """Test when reply_to_message returns success=False."""
        mock_connector.reply_to_message = AsyncMock(return_value={"success": False})
        with patch(
            "aragora.server.handlers.features.outlook._get_or_create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handle_reply_message.__wrapped__(
                workspace_id="default",
                user_id="user1",
                message_id="msg-1",
                body="Thanks!",
            )
        assert result["success"] is False
