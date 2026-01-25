"""
Tests for Shared Inbox Handler.

Tests cover:
- Handler routing for inbox and routing rules endpoints
- Input validation for create/update operations
- Workspace ID requirements
- Error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.server.handlers.shared_inbox import (
    SharedInboxHandler,
    MessageStatus,
    SharedInbox,
    SharedInboxMessage,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return {"user_store": MagicMock(), "nomic_dir": "/tmp/test"}


@pytest.fixture
def handler(mock_server_context):
    """Create SharedInboxHandler with mock context."""
    h = SharedInboxHandler(mock_server_context)
    # Mock _get_user_id to return a test user
    h._get_user_id = MagicMock(return_value="test-user-123")
    return h


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    return mock


# ============================================================================
# Routing Tests
# ============================================================================


class TestSharedInboxHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_shared_inbox_base(self, handler):
        """Handler can handle base shared inbox endpoint."""
        assert handler.can_handle("/api/v1/inbox/shared")

    def test_can_handle_shared_inbox_with_id(self, handler):
        """Handler can handle shared inbox with ID."""
        assert handler.can_handle("/api/v1/inbox/shared/inbox-123")
        assert handler.can_handle("/api/v1/inbox/shared/uuid-1234-5678")

    def test_can_handle_shared_inbox_messages(self, handler):
        """Handler can handle inbox messages endpoint."""
        assert handler.can_handle("/api/v1/inbox/shared/inbox-123/messages")

    def test_can_handle_message_assign(self, handler):
        """Handler can handle message assignment endpoint."""
        assert handler.can_handle("/api/v1/inbox/shared/inbox-123/messages/msg-456/assign")

    def test_can_handle_message_status(self, handler):
        """Handler can handle message status endpoint."""
        assert handler.can_handle("/api/v1/inbox/shared/inbox-123/messages/msg-456/status")

    def test_can_handle_message_tag(self, handler):
        """Handler can handle message tag endpoint."""
        assert handler.can_handle("/api/v1/inbox/shared/inbox-123/messages/msg-456/tag")

    def test_can_handle_routing_rules_base(self, handler):
        """Handler can handle routing rules base endpoint."""
        assert handler.can_handle("/api/v1/inbox/routing/rules")

    def test_can_handle_routing_rules_with_id(self, handler):
        """Handler can handle routing rules with ID."""
        assert handler.can_handle("/api/v1/inbox/routing/rules/rule-123")

    def test_can_handle_routing_rules_test(self, handler):
        """Handler can handle routing rule test endpoint."""
        assert handler.can_handle("/api/v1/inbox/routing/rules/rule-123/test")

    def test_cannot_handle_unknown_path(self, handler):
        """Handler cannot handle unknown paths."""
        assert not handler.can_handle("/api/v1/other/endpoint")
        assert not handler.can_handle("/api/v1/inbox/other")
        assert not handler.can_handle("/api/v1/messages")


# ============================================================================
# Data Model Tests
# ============================================================================


class TestMessageStatus:
    """Tests for MessageStatus enum."""

    def test_message_status_values(self):
        """MessageStatus has expected values."""
        assert MessageStatus.OPEN.value == "open"
        assert MessageStatus.ASSIGNED.value == "assigned"
        assert MessageStatus.RESOLVED.value == "resolved"
        assert MessageStatus.CLOSED.value == "closed"


class TestSharedInboxMessage:
    """Tests for SharedInboxMessage dataclass."""

    def test_create_message(self):
        """Can create SharedInboxMessage."""
        from datetime import datetime, timezone

        msg = SharedInboxMessage(
            id="msg-123",
            inbox_id="inbox-456",
            email_id="email-789",
            subject="Test Subject",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            snippet="Test content...",
            received_at=datetime.now(timezone.utc),
        )
        assert msg.id == "msg-123"
        assert msg.inbox_id == "inbox-456"
        assert msg.status == MessageStatus.OPEN  # Default
        assert msg.tags == []
        assert msg.assigned_to is None

    def test_message_with_status(self):
        """Can create message with specific status."""
        from datetime import datetime, timezone

        msg = SharedInboxMessage(
            id="msg-123",
            inbox_id="inbox-456",
            email_id="email-789",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            snippet="Content...",
            received_at=datetime.now(timezone.utc),
            status=MessageStatus.ASSIGNED,
            assigned_to="user-789",
        )
        assert msg.status == MessageStatus.ASSIGNED
        assert msg.assigned_to == "user-789"


class TestSharedInbox:
    """Tests for SharedInbox dataclass."""

    def test_create_inbox(self):
        """Can create SharedInbox."""
        inbox = SharedInbox(
            id="inbox-123",
            workspace_id="ws-456",
            name="Support Inbox",
        )
        assert inbox.id == "inbox-123"
        assert inbox.workspace_id == "ws-456"
        assert inbox.name == "Support Inbox"
        assert inbox.team_members == []
        assert inbox.admins == []

    def test_inbox_with_team(self):
        """Can create inbox with team members."""
        inbox = SharedInbox(
            id="inbox-123",
            workspace_id="ws-456",
            name="Support",
            team_members=["user-1", "user-2"],
            admins=["admin-1"],
        )
        assert len(inbox.team_members) == 2
        assert len(inbox.admins) == 1


# ============================================================================
# Handler Method Tests
# ============================================================================


class TestHandlePostSharedInbox:
    """Tests for POST /api/v1/inbox/shared."""

    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler):
        """Returns error when workspace_id is missing."""
        result = await handler.handle_post_shared_inbox(
            {
                "name": "Test Inbox",
            }
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_name(self, handler):
        """Returns error when name is missing."""
        result = await handler.handle_post_shared_inbox(
            {
                "workspace_id": "ws-123",
            }
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_create(self, handler):
        """Creates inbox with valid data."""
        with patch(
            "aragora.server.handlers.shared_inbox.handle_create_shared_inbox",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = {
                "success": True,
                "inbox": {"inbox_id": "inbox-123", "name": "Test"},
            }

            result = await handler.handle_post_shared_inbox(
                {
                    "workspace_id": "ws-123",
                    "name": "Test Inbox",
                }
            )
            assert result.status_code == 200


class TestHandleGetSharedInboxes:
    """Tests for GET /api/v1/inbox/shared."""

    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler):
        """Returns error when workspace_id is missing."""
        result = await handler.handle_get_shared_inboxes({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_list(self, handler):
        """Lists inboxes with valid workspace_id."""
        with patch(
            "aragora.server.handlers.shared_inbox.handle_list_shared_inboxes",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = {
                "success": True,
                "inboxes": [],
            }

            result = await handler.handle_get_shared_inboxes(
                {
                    "workspace_id": "ws-123",
                }
            )
            assert result.status_code == 200


class TestHandleGetSharedInbox:
    """Tests for GET /api/v1/inbox/shared/:id."""

    @pytest.mark.asyncio
    async def test_get_inbox_success(self, handler):
        """Gets inbox details successfully."""
        with patch(
            "aragora.server.handlers.shared_inbox.handle_get_shared_inbox",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = {
                "success": True,
                "inbox": {"inbox_id": "inbox-123", "name": "Test"},
            }

            result = await handler.handle_get_shared_inbox({}, "inbox-123")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_inbox_not_found(self, handler):
        """Returns 404 when inbox not found."""
        with patch(
            "aragora.server.handlers.shared_inbox.handle_get_shared_inbox",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = {
                "success": False,
                "error": "Inbox not found",
            }

            result = await handler.handle_get_shared_inbox({}, "nonexistent")
            assert result.status_code == 404


# ============================================================================
# Handler Initialization Tests
# ============================================================================


class TestSharedInboxHandlerInit:
    """Tests for handler initialization."""

    def test_handler_has_routes(self, handler):
        """Handler has ROUTES list."""
        assert len(handler.ROUTES) >= 2

    def test_handler_has_route_prefixes(self, handler):
        """Handler has ROUTE_PREFIXES list."""
        assert len(handler.ROUTE_PREFIXES) >= 2

    def test_handler_extends_base_handler(self, handler):
        """Handler extends BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)
