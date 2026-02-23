"""Tests for workspace invite handler (WorkspaceInvitesMixin).

Tests the invite management endpoints:
- POST   /api/v1/workspaces/{workspace_id}/invites              - Create invite
- GET    /api/v1/workspaces/{workspace_id}/invites              - List invites
- DELETE /api/v1/workspaces/{workspace_id}/invites/{invite_id}  - Cancel invite
- POST   /api/v1/workspaces/{workspace_id}/invites/{invite_id}/resend - Resend invite
- POST   /api/v1/invites/{token}/accept                        - Accept invite
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract JSON body from HandlerResult."""
    try:
        return json.loads(result.body.decode("utf-8"))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return {}


def _error(result) -> str:
    """Extract error message from HandlerResult."""
    body = _body(result)
    return body.get("error", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_invite_store():
    """Reset the global invite store before each test."""
    from aragora.server.handlers.workspace.invites import _invite_store_lazy

    _invite_store_lazy._store = None
    _invite_store_lazy._initialized = False
    _invite_store_lazy._init_error = None
    yield
    _invite_store_lazy._store = None
    _invite_store_lazy._initialized = False
    _invite_store_lazy._init_error = None


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state between tests."""
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass


@pytest.fixture
def mock_workspace_module():
    """Patch the workspace_module imported via _mod().

    Returns a MagicMock that provides all the symbols the invite mixin
    accesses through ``_mod()``.
    """
    m = MagicMock()

    # Auth context returned by extract_user_from_request
    auth_ctx = MagicMock()
    auth_ctx.is_authenticated = True
    auth_ctx.user_id = "test-user-001"
    m.extract_user_from_request.return_value = auth_ctx

    # Permission constants
    m.PERM_WORKSPACE_SHARE = "workspace:share"
    m.PERM_WORKSPACE_READ = "workspace:read"

    # WorkspacePermission enum
    m.WorkspacePermission = MagicMock()
    m.WorkspacePermission.READ = "read"
    m.WorkspacePermission.WRITE = "write"
    m.WorkspacePermission.ADMIN = "admin"

    # Audit types
    m.AuditAction = MagicMock()
    m.AuditAction.WRITE = "write"
    m.AuditAction.DELETE = "delete"
    m.AuditAction.ADD_MEMBER = "add_member"
    m.AuditOutcome = MagicMock()
    m.AuditOutcome.SUCCESS = "success"
    m.Actor = MagicMock()
    m.Resource = MagicMock()

    # AccessDeniedException
    m.AccessDeniedException = type("AccessDeniedException", (Exception,), {})

    # Response helpers -- delegate to real implementations
    from aragora.server.handlers.base import json_response, error_response

    m.json_response = json_response
    m.error_response = error_response

    with patch(
        "aragora.server.handlers.workspace.invites._mod",
        return_value=m,
    ):
        yield m


@pytest.fixture
def handler(mock_workspace_module):
    """Create a WorkspaceInvitesMixin instance with mocked dependencies."""
    from aragora.server.handlers.workspace.invites import WorkspaceInvitesMixin

    class _TestHandler(WorkspaceInvitesMixin):
        """Concrete handler combining mixin with mock infrastructure."""

        def _get_user_store(self):
            return MagicMock()

        def _get_isolation_manager(self):
            manager = MagicMock()
            manager.get_workspace = AsyncMock()
            manager.add_member = AsyncMock()
            return manager

        def _get_audit_log(self):
            audit = MagicMock()
            audit.log = AsyncMock()
            return audit

        def _run_async(self, coro):
            """Run coroutine synchronously for tests."""
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're inside an event loop already -- create a task trick
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(asyncio.run, coro).result()
                return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)

        def _check_rbac_permission(self, handler, perm, auth_ctx):
            return None  # Always allow in tests

        def read_json_body(self, handler):
            return getattr(handler, "_json_body", None)

    return _TestHandler()


@pytest.fixture
def make_handler_request():
    """Factory for creating mock HTTP handler objects."""

    def _make(
        path: str = "/api/v1/workspaces/ws-1/invites",
        method: str = "POST",
        body: dict[str, Any] | None = None,
        query: str = "",
    ):
        h = MagicMock()
        full_path = f"{path}?{query}" if query else path
        h.path = full_path
        h.command = method
        h.headers = {"Content-Length": "0"}
        if body is not None:
            h._json_body = body
        else:
            h._json_body = None
        return h

    return _make


# ---------------------------------------------------------------------------
# InviteStore unit tests
# ---------------------------------------------------------------------------


class TestInviteStore:
    """Test InviteStore in-memory operations."""

    def test_create_invite(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "user@example.com", "member", "creator-1")
        assert invite.workspace_id == "ws-1"
        assert invite.email == "user@example.com"
        assert invite.role == "member"
        assert invite.created_by == "creator-1"
        assert invite.id.startswith("inv_")

    def test_create_normalizes_email(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "  User@EXAMPLE.com  ", "member", "creator-1")
        assert invite.email == "user@example.com"

    def test_get_invite_by_id(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        assert store.get(invite.id) is invite

    def test_get_nonexistent_returns_none(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        assert store.get("inv_nonexistent") is None

    def test_get_by_token(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        found = store.get_by_token(invite.token)
        assert found is invite

    def test_get_by_invalid_token_returns_none(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        assert store.get_by_token("bad-token") is None

    def test_list_for_workspace(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        store.create("ws-1", "a@b.com", "member", "c")
        store.create("ws-1", "d@e.com", "admin", "c")
        store.create("ws-2", "f@g.com", "viewer", "c")
        result = store.list_for_workspace("ws-1")
        assert len(result) == 2

    def test_list_for_workspace_with_status_filter(self):
        from aragora.server.handlers.workspace.invites import (
            InviteStore,
            InviteStatus,
        )

        store = InviteStore()
        inv1 = store.create("ws-1", "a@b.com", "member", "c")
        store.create("ws-1", "d@e.com", "admin", "c")
        store.update_status(inv1.id, InviteStatus.CANCELED)
        pending = store.list_for_workspace("ws-1", InviteStatus.PENDING)
        assert len(pending) == 1

    def test_list_for_workspace_sorted_by_created_at_desc(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        inv1 = store.create("ws-1", "a@b.com", "member", "c")
        inv2 = store.create("ws-1", "d@e.com", "admin", "c")
        result = store.list_for_workspace("ws-1")
        assert result[0].id == inv2.id  # Most recent first

    def test_update_status(self):
        from aragora.server.handlers.workspace.invites import (
            InviteStore,
            InviteStatus,
        )

        store = InviteStore()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        assert store.update_status(invite.id, InviteStatus.CANCELED)
        assert invite.status == InviteStatus.CANCELED

    def test_update_status_with_accepted_by(self):
        from aragora.server.handlers.workspace.invites import (
            InviteStore,
            InviteStatus,
        )

        store = InviteStore()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(invite.id, InviteStatus.ACCEPTED, accepted_by="joiner")
        assert invite.accepted_by == "joiner"
        assert invite.accepted_at is not None

    def test_update_status_nonexistent(self):
        from aragora.server.handlers.workspace.invites import (
            InviteStore,
            InviteStatus,
        )

        store = InviteStore()
        assert not store.update_status("inv_nope", InviteStatus.CANCELED)

    def test_delete_invite(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        token = invite.token
        assert store.delete(invite.id)
        assert store.get(invite.id) is None
        assert store.get_by_token(token) is None

    def test_delete_nonexistent(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        assert not store.delete("inv_nope")

    def test_check_existing_finds_pending(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "user@example.com", "member", "c")
        found = store.check_existing("ws-1", "user@example.com")
        assert found is invite

    def test_check_existing_normalizes_email(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "user@example.com", "member", "c")
        found = store.check_existing("ws-1", "  USER@Example.COM  ")
        assert found is invite

    def test_check_existing_ignores_canceled(self):
        from aragora.server.handlers.workspace.invites import (
            InviteStore,
            InviteStatus,
        )

        store = InviteStore()
        invite = store.create("ws-1", "user@example.com", "member", "c")
        store.update_status(invite.id, InviteStatus.CANCELED)
        assert store.check_existing("ws-1", "user@example.com") is None

    def test_check_existing_ignores_expired(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "user@example.com", "member", "c", expires_in_days=0)
        # Manually expire it
        invite.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert store.check_existing("ws-1", "user@example.com") is None

    def test_check_existing_different_workspace(self):
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        store.create("ws-1", "user@example.com", "member", "c")
        assert store.check_existing("ws-2", "user@example.com") is None


# ---------------------------------------------------------------------------
# WorkspaceInvite dataclass tests
# ---------------------------------------------------------------------------


class TestWorkspaceInvite:
    """Test WorkspaceInvite model methods."""

    def test_is_valid_pending_not_expired(self):
        from aragora.server.handlers.workspace.invites import (
            WorkspaceInvite,
            InviteStatus,
        )

        invite = WorkspaceInvite(
            id="inv_1",
            workspace_id="ws-1",
            email="a@b.com",
            role="member",
            token="tok",
            status=InviteStatus.PENDING,
            created_by="c",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )
        assert invite.is_valid()

    def test_is_valid_expired(self):
        from aragora.server.handlers.workspace.invites import (
            WorkspaceInvite,
            InviteStatus,
        )

        invite = WorkspaceInvite(
            id="inv_1",
            workspace_id="ws-1",
            email="a@b.com",
            role="member",
            token="tok",
            status=InviteStatus.PENDING,
            created_by="c",
            created_at=datetime.now(timezone.utc) - timedelta(days=8),
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert not invite.is_valid()

    def test_is_valid_not_pending(self):
        from aragora.server.handlers.workspace.invites import (
            WorkspaceInvite,
            InviteStatus,
        )

        invite = WorkspaceInvite(
            id="inv_1",
            workspace_id="ws-1",
            email="a@b.com",
            role="member",
            token="tok",
            status=InviteStatus.ACCEPTED,
            created_by="c",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )
        assert not invite.is_valid()

    def test_to_dict_keys(self):
        from aragora.server.handlers.workspace.invites import (
            WorkspaceInvite,
            InviteStatus,
        )

        invite = WorkspaceInvite(
            id="inv_1",
            workspace_id="ws-1",
            email="a@b.com",
            role="member",
            token="tok",
            status=InviteStatus.PENDING,
            created_by="c",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )
        d = invite.to_dict()
        assert d["id"] == "inv_1"
        assert d["workspace_id"] == "ws-1"
        assert d["email"] == "a@b.com"
        assert d["role"] == "member"
        assert d["status"] == "pending"
        assert d["created_by"] == "c"
        assert "created_at" in d
        assert "expires_at" in d
        assert d["accepted_by"] is None
        assert d["accepted_at"] is None

    def test_to_dict_with_accepted(self):
        from aragora.server.handlers.workspace.invites import (
            WorkspaceInvite,
            InviteStatus,
        )

        now = datetime.now(timezone.utc)
        invite = WorkspaceInvite(
            id="inv_1",
            workspace_id="ws-1",
            email="a@b.com",
            role="member",
            token="tok",
            status=InviteStatus.ACCEPTED,
            created_by="c",
            created_at=now,
            expires_at=now + timedelta(days=7),
            accepted_by="joiner",
            accepted_at=now,
        )
        d = invite.to_dict()
        assert d["accepted_by"] == "joiner"
        assert d["accepted_at"] is not None


# ---------------------------------------------------------------------------
# InviteStatus tests
# ---------------------------------------------------------------------------


class TestInviteStatus:
    """Test InviteStatus enum."""

    def test_values(self):
        from aragora.server.handlers.workspace.invites import InviteStatus

        assert InviteStatus.PENDING.value == "pending"
        assert InviteStatus.ACCEPTED.value == "accepted"
        assert InviteStatus.EXPIRED.value == "expired"
        assert InviteStatus.CANCELED.value == "canceled"

    def test_is_string_enum(self):
        from aragora.server.handlers.workspace.invites import InviteStatus

        assert isinstance(InviteStatus.PENDING, str)


# ---------------------------------------------------------------------------
# POST /api/v1/workspaces/{workspace_id}/invites - Create Invite
# ---------------------------------------------------------------------------


class TestCreateInvite:
    """Test the _handle_create_invite endpoint."""

    def test_create_success(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "new@example.com"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201
        body = _body(result)
        assert "email_masked" in body
        assert "invite_url" in body
        assert body["role"] == "member"  # default role
        assert body["status"] == "pending"

    def test_create_with_role(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "new@example.com", "role": "admin"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201
        assert _body(result)["role"] == "admin"

    def test_create_with_owner_role(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "new@example.com", "role": "owner"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201
        assert _body(result)["role"] == "owner"

    def test_create_with_viewer_role(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "new@example.com", "role": "viewer"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201
        assert _body(result)["role"] == "viewer"

    def test_create_with_member_role(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "new@example.com", "role": "member"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201
        assert _body(result)["role"] == "member"

    def test_create_invalid_role(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "new@example.com", "role": "superadmin"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 400
        assert "Invalid role" in _error(result)

    def test_create_missing_email(self, handler, make_handler_request):
        req = make_handler_request(body={})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 400
        assert "email" in _error(result).lower()

    def test_create_empty_email(self, handler, make_handler_request):
        req = make_handler_request(body={"email": ""})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 400

    def test_create_invalid_email_no_at(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "notanemail"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 400
        assert "email" in _error(result).lower()

    def test_create_null_json_body(self, handler, make_handler_request):
        req = make_handler_request(body=None)
        # read_json_body returns None -> "Invalid JSON body"
        req._json_body = None
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 400
        assert "JSON" in _error(result)

    def test_create_email_masked_format(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "alice@example.com"})
        result = handler._handle_create_invite(req, "ws-1")
        body = _body(result)
        assert body["email_masked"] == "al***@example.com"

    def test_create_invite_url_contains_token(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "new@example.com"})
        result = handler._handle_create_invite(req, "ws-1")
        body = _body(result)
        assert body["invite_url"].startswith("/invites/")
        assert body["invite_url"].endswith("/accept")

    def test_create_custom_expires_in_days(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "new@example.com", "expires_in_days": 14})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201

    def test_create_duplicate_pending_invite(self, handler, make_handler_request):
        req1 = make_handler_request(body={"email": "dup@example.com"})
        result1 = handler._handle_create_invite(req1, "ws-1")
        assert _status(result1) == 201

        req2 = make_handler_request(body={"email": "dup@example.com"})
        result2 = handler._handle_create_invite(req2, "ws-1")
        assert _status(result2) == 409
        assert "pending invite already exists" in _error(result2).lower()

    def test_create_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False
        req = make_handler_request(body={"email": "new@example.com"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 401

    def test_create_workspace_access_denied(
        self, handler, make_handler_request, mock_workspace_module
    ):
        exc_class = mock_workspace_module.AccessDeniedException

        mgr = MagicMock()
        mgr.get_workspace = AsyncMock(side_effect=exc_class("denied"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(body={"email": "new@example.com"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 403

    def test_create_different_workspace_allows_same_email(self, handler, make_handler_request):
        req1 = make_handler_request(body={"email": "same@example.com"})
        result1 = handler._handle_create_invite(req1, "ws-1")
        assert _status(result1) == 201

        req2 = make_handler_request(body={"email": "same@example.com"})
        result2 = handler._handle_create_invite(req2, "ws-2")
        assert _status(result2) == 201


# ---------------------------------------------------------------------------
# GET /api/v1/workspaces/{workspace_id}/invites - List Invites
# ---------------------------------------------------------------------------


class TestListInvites:
    """Test the _handle_list_invites endpoint."""

    def test_list_empty(self, handler, make_handler_request):
        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-1"
        assert body["invites"] == []
        assert body["total"] == 0

    def test_list_with_invites(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        store.create("ws-1", "a@b.com", "member", "c")
        store.create("ws-1", "d@e.com", "admin", "c")

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["invites"]) == 2

    def test_list_filters_by_workspace(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        store.create("ws-1", "a@b.com", "member", "c")
        store.create("ws-2", "d@e.com", "admin", "c")

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _body(result)["total"] == 1

    def test_list_with_status_filter_pending(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        inv1 = store.create("ws-1", "a@b.com", "member", "c")
        store.create("ws-1", "d@e.com", "admin", "c")
        store.update_status(inv1.id, InviteStatus.CANCELED)

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
            query="status=pending",
        )
        result = handler._handle_list_invites(req, "ws-1")
        body = _body(result)
        assert body["total"] == 1

    def test_list_with_status_filter_canceled(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        inv1 = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(inv1.id, InviteStatus.CANCELED)

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
            query="status=canceled",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _body(result)["total"] == 1

    def test_list_invalid_status_filter(self, handler, make_handler_request):
        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
            query="status=bogus",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _status(result) == 400
        assert "Invalid status" in _error(result)

    def test_list_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False
        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _status(result) == 401

    def test_list_workspace_access_denied(
        self, handler, make_handler_request, mock_workspace_module
    ):
        exc_class = mock_workspace_module.AccessDeniedException
        mgr = MagicMock()
        mgr.get_workspace = AsyncMock(side_effect=exc_class("denied"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _status(result) == 403

    def test_list_marks_expired_invites(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c", expires_in_days=0)
        # Force expiration
        invite.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
        )
        result = handler._handle_list_invites(req, "ws-1")
        body = _body(result)
        assert body["invites"][0]["status"] == "expired"


# ---------------------------------------------------------------------------
# DELETE /api/v1/workspaces/{workspace_id}/invites/{invite_id} - Cancel Invite
# ---------------------------------------------------------------------------


class TestCancelInvite:
    """Test the _handle_cancel_invite endpoint."""

    def test_cancel_success(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-1", invite.id)
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Invite canceled"
        assert body["invite_id"] == invite.id

    def test_cancel_not_found(self, handler, make_handler_request):
        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-1", "inv_nonexistent")
        assert _status(result) == 404
        assert "not found" in _error(result).lower()

    def test_cancel_wrong_workspace(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-OTHER", invite.id)
        assert _status(result) == 404
        assert "not found in this workspace" in _error(result).lower()

    def test_cancel_already_canceled(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(invite.id, InviteStatus.CANCELED)

        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-1", invite.id)
        assert _status(result) == 400
        assert "canceled" in _error(result).lower()

    def test_cancel_accepted_invite(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(invite.id, InviteStatus.ACCEPTED, accepted_by="user")

        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-1", invite.id)
        assert _status(result) == 400
        assert "accepted" in _error(result).lower()

    def test_cancel_expired_invite(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(invite.id, InviteStatus.EXPIRED)

        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-1", invite.id)
        assert _status(result) == 400
        assert "expired" in _error(result).lower()

    def test_cancel_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False
        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-1", "inv_any")
        assert _status(result) == 401


# ---------------------------------------------------------------------------
# POST /api/v1/workspaces/{ws}/invites/{id}/resend - Resend Invite
# ---------------------------------------------------------------------------


class TestResendInvite:
    """Test the _handle_resend_invite endpoint."""

    def test_resend_success(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        old_expires = invite.expires_at

        req = make_handler_request(method="POST")
        result = handler._handle_resend_invite(req, "ws-1", invite.id)
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Invite resent"
        assert body["invite_id"] == invite.id
        assert "new_expires_at" in body
        # New expiration should be later than original
        assert invite.expires_at >= old_expires

    def test_resend_not_found(self, handler, make_handler_request):
        req = make_handler_request(method="POST")
        result = handler._handle_resend_invite(req, "ws-1", "inv_nonexistent")
        assert _status(result) == 404

    def test_resend_wrong_workspace(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        req = make_handler_request(method="POST")
        result = handler._handle_resend_invite(req, "ws-OTHER", invite.id)
        assert _status(result) == 404

    def test_resend_canceled_invite(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(invite.id, InviteStatus.CANCELED)

        req = make_handler_request(method="POST")
        result = handler._handle_resend_invite(req, "ws-1", invite.id)
        assert _status(result) == 400
        assert "canceled" in _error(result).lower()

    def test_resend_accepted_invite(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(invite.id, InviteStatus.ACCEPTED, accepted_by="user")

        req = make_handler_request(method="POST")
        result = handler._handle_resend_invite(req, "ws-1", invite.id)
        assert _status(result) == 400

    def test_resend_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False
        req = make_handler_request(method="POST")
        result = handler._handle_resend_invite(req, "ws-1", "inv_any")
        assert _status(result) == 401


# ---------------------------------------------------------------------------
# POST /api/v1/invites/{token}/accept - Accept Invite
# ---------------------------------------------------------------------------


class TestAcceptInvite:
    """Test the _handle_accept_invite endpoint."""

    def test_accept_success(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Successfully joined workspace"
        assert body["workspace_id"] == "ws-1"
        assert body["role"] == "member"

    def test_accept_marks_invite_accepted(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        req = make_handler_request(method="POST")
        handler._handle_accept_invite(req, invite.token)
        assert invite.status == InviteStatus.ACCEPTED
        assert invite.accepted_by == "test-user-001"

    def test_accept_invalid_token(self, handler, make_handler_request):
        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, "invalid-token")
        assert _status(result) == 404
        assert "Invalid or expired" in _error(result)

    def test_accept_expired_invite(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c", expires_in_days=0)
        invite.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 410
        assert "expired" in _error(result).lower()

    def test_accept_canceled_invite(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(invite.id, InviteStatus.CANCELED)

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 410
        assert "canceled" in _error(result).lower()

    def test_accept_already_accepted(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(invite.id, InviteStatus.ACCEPTED, accepted_by="prev-user")

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 410
        assert "accepted" in _error(result).lower()

    def test_accept_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False
        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, "any-token")
        assert _status(result) == 401

    def test_accept_add_member_access_denied(
        self, handler, make_handler_request, mock_workspace_module
    ):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        exc_class = mock_workspace_module.AccessDeniedException
        mgr = MagicMock()
        mgr.add_member = AsyncMock(side_effect=exc_class("denied"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 403

    def test_accept_add_member_error(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock(side_effect=ValueError("DB error"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 500

    def test_accept_admin_role_permissions(self, handler, make_handler_request):
        """Admin role should grant READ, WRITE, ADMIN permissions."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "admin", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock()
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 200
        # Check that add_member was called with 3 permissions
        call_args = mgr.add_member.call_args
        assert len(call_args.kwargs.get("permissions", call_args[1].get("permissions", []))) == 3

    def test_accept_viewer_role_permissions(self, handler, make_handler_request):
        """Viewer role should grant only READ permission."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "viewer", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock()
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 200
        call_args = mgr.add_member.call_args
        assert len(call_args.kwargs.get("permissions", call_args[1].get("permissions", []))) == 1

    def test_accept_member_role_permissions(self, handler, make_handler_request):
        """Member role should grant READ, WRITE permissions."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock()
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 200
        call_args = mgr.add_member.call_args
        assert len(call_args.kwargs.get("permissions", call_args[1].get("permissions", []))) == 2

    def test_accept_owner_role_permissions(self, handler, make_handler_request):
        """Owner role should grant READ, WRITE, ADMIN permissions."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "owner", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock()
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 200
        call_args = mgr.add_member.call_args
        assert len(call_args.kwargs.get("permissions", call_args[1].get("permissions", []))) == 3


# ---------------------------------------------------------------------------
# get_invite_store global accessor
# ---------------------------------------------------------------------------


class TestGetInviteStore:
    """Test the global store accessor."""

    def test_returns_invite_store_instance(self):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStore,
        )

        store = get_invite_store()
        assert isinstance(store, InviteStore)

    def test_returns_same_instance(self):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store1 = get_invite_store()
        store2 = get_invite_store()
        assert store1 is store2


# ---------------------------------------------------------------------------
# Module __all__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports(self):
        from aragora.server.handlers.workspace import invites

        assert "WorkspaceInvitesMixin" in invites.__all__
        assert "InviteStore" in invites.__all__
        assert "WorkspaceInvite" in invites.__all__
        assert "InviteStatus" in invites.__all__
        assert "get_invite_store" in invites.__all__

    def test_all_count(self):
        from aragora.server.handlers.workspace import invites

        assert len(invites.__all__) == 5


# ---------------------------------------------------------------------------
# RBAC permission check integration
# ---------------------------------------------------------------------------


class TestRBACPermissionCheck:
    """Test RBAC permission gating on handler methods."""

    def test_create_checks_rbac(self, handler, make_handler_request, mock_workspace_module):
        """When _check_rbac_permission returns an error, create should return it."""
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)
        req = make_handler_request(body={"email": "x@y.com"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 403

    def test_cancel_checks_rbac(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)
        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-1", "inv_any")
        assert _status(result) == 403

    def test_resend_checks_rbac(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)
        req = make_handler_request(method="POST")
        result = handler._handle_resend_invite(req, "ws-1", "inv_any")
        assert _status(result) == 403

    def test_list_checks_rbac(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)
        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _status(result) == 403


# ---------------------------------------------------------------------------
# Edge cases and additional coverage
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_create_email_with_plus_addressing(self, handler, make_handler_request):
        """Email with + addressing should be accepted."""
        req = make_handler_request(body={"email": "user+tag@example.com"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201

    def test_create_email_with_subdomain(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "user@sub.example.com"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201

    def test_create_invite_workspace_id_preserved(self, handler, make_handler_request):
        req = make_handler_request(body={"email": "a@b.com"})
        result = handler._handle_create_invite(req, "ws-special-123")
        assert _status(result) == 201
        assert _body(result)["workspace_id"] == "ws-special-123"

    def test_store_create_with_zero_expiry(self):
        """Invite with 0 day expiry is created but expires immediately."""
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "a@b.com", "member", "c", expires_in_days=0)
        # expires_at should be within seconds of created_at
        assert abs((invite.expires_at - invite.created_at).total_seconds()) < 5

    def test_invite_to_dict_has_no_token(self):
        """The to_dict representation should NOT include the token."""
        from aragora.server.handlers.workspace.invites import InviteStore

        store = InviteStore()
        invite = store.create("ws-1", "a@b.com", "member", "c")
        d = invite.to_dict()
        assert "token" not in d

    def test_list_no_status_query_returns_all(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        store.create("ws-1", "a@b.com", "member", "c")
        inv2 = store.create("ws-1", "d@e.com", "admin", "c")
        store.update_status(inv2.id, InviteStatus.CANCELED)

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
        )
        result = handler._handle_list_invites(req, "ws-1")
        assert _body(result)["total"] == 2  # Both returned without filter

    def test_accept_unknown_role_falls_back_to_read(self, handler, make_handler_request):
        """If the invite has an unexpected role, fallback to READ permission."""
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "custom_role", "c")
        # The store allows any role string; permission mapping falls back

        mgr = MagicMock()
        mgr.add_member = AsyncMock()
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 200
        call_args = mgr.add_member.call_args
        perms = call_args.kwargs.get("permissions", call_args[1].get("permissions", []))
        # Fallback is [WorkspacePermission.READ] -> 1 item
        assert len(perms) == 1

    def test_accept_add_member_called_with_correct_args(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "creator-user")

        mgr = MagicMock()
        mgr.add_member = AsyncMock()
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        handler._handle_accept_invite(req, invite.token)

        mgr.add_member.assert_called_once()
        call_kwargs = mgr.add_member.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-1"
        assert call_kwargs["user_id"] == "test-user-001"
        assert call_kwargs["added_by"] == "creator-user"

    def test_create_audit_log_called(self, handler, make_handler_request):
        """Creating an invite should log to audit."""
        audit = MagicMock()
        audit.log = AsyncMock()
        handler._get_audit_log = lambda: audit

        req = make_handler_request(body={"email": "audit@test.com"})
        result = handler._handle_create_invite(req, "ws-1")
        assert _status(result) == 201
        audit.log.assert_called_once()

    def test_cancel_audit_log_called(self, handler, make_handler_request):
        """Canceling an invite should log to audit."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        audit = MagicMock()
        audit.log = AsyncMock()
        handler._get_audit_log = lambda: audit

        req = make_handler_request(method="DELETE")
        result = handler._handle_cancel_invite(req, "ws-1", invite.id)
        assert _status(result) == 200
        audit.log.assert_called_once()

    def test_accept_audit_log_called(self, handler, make_handler_request):
        """Accepting an invite should log to audit."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        audit = MagicMock()
        audit.log = AsyncMock()
        handler._get_audit_log = lambda: audit

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 200
        audit.log.assert_called_once()

    def test_accept_runtime_error_returns_500(self, handler, make_handler_request):
        """RuntimeError from add_member returns 500."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock(side_effect=RuntimeError("bad"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 500

    def test_accept_os_error_returns_500(self, handler, make_handler_request):
        """OSError from add_member returns 500."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock(side_effect=OSError("io"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 500

    def test_accept_type_error_returns_500(self, handler, make_handler_request):
        """TypeError from add_member returns 500."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock(side_effect=TypeError("type"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 500

    def test_accept_key_error_returns_500(self, handler, make_handler_request):
        """KeyError from add_member returns 500."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock(side_effect=KeyError("missing"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 500

    def test_accept_attribute_error_returns_500(self, handler, make_handler_request):
        """AttributeError from add_member returns 500."""
        from aragora.server.handlers.workspace.invites import get_invite_store

        store = get_invite_store()
        invite = store.create("ws-1", "a@b.com", "member", "c")

        mgr = MagicMock()
        mgr.add_member = AsyncMock(side_effect=AttributeError("attr"))
        handler._get_isolation_manager = lambda: mgr

        req = make_handler_request(method="POST")
        result = handler._handle_accept_invite(req, invite.token)
        assert _status(result) == 500

    def test_list_status_filter_accepted(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        inv = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(inv.id, InviteStatus.ACCEPTED, accepted_by="u")

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
            query="status=accepted",
        )
        result = handler._handle_list_invites(req, "ws-1")
        body = _body(result)
        assert body["total"] == 1
        assert body["invites"][0]["status"] == "accepted"

    def test_list_status_filter_expired(self, handler, make_handler_request):
        from aragora.server.handlers.workspace.invites import (
            get_invite_store,
            InviteStatus,
        )

        store = get_invite_store()
        inv = store.create("ws-1", "a@b.com", "member", "c")
        store.update_status(inv.id, InviteStatus.EXPIRED)

        req = make_handler_request(
            path="/api/v1/workspaces/ws-1/invites",
            method="GET",
            query="status=expired",
        )
        result = handler._handle_list_invites(req, "ws-1")
        body = _body(result)
        assert body["total"] == 1
        assert body["invites"][0]["status"] == "expired"
