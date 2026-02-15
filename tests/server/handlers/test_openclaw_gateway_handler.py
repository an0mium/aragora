"""
Comprehensive tests for OpenClawGatewayHandler - OpenClaw gateway HTTP endpoints.

Tests cover:
1. Session creation and retrieval
2. Session listing with filters
3. Action execution
4. Action status and cancellation
5. Credential storage (no secret in response)
6. Credential listing and deletion
7. Credential rotation
8. Health endpoint
9. Metrics endpoint
10. Audit log with filters
11. RBAC permission enforcement
12. Error handling
13. Path normalization and routing (v1 paths, alternate prefixes)
14. Admin user access overrides
15. End session via POST
16. Policy and approval endpoints
17. Stats endpoint
18. Cross-ownership access control
19. Data model edge cases
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.openclaw_gateway import (
    Action,
    ActionStatus,
    AuditEntry,
    Credential,
    CredentialType,
    OpenClawGatewayHandler,
    OpenClawGatewayStore,
    Session,
    SessionStatus,
    _get_store,
    get_openclaw_gateway_handler,
)


# ===========================================================================
# Test Fixtures and Helpers
# ===========================================================================


@dataclass
class MockUser:
    """Mock user authentication context."""

    user_id: str = "user-001"
    email: str = "test@example.com"
    org_id: str | None = "org-001"
    role: str = "user"
    permissions: list[str] = field(default_factory=list)
    is_authenticated: bool = True


class MockRequestHandler:
    """Mock HTTP request handler."""

    def __init__(
        self,
        body: dict | None = None,
        headers: dict | None = None,
    ):
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}
        if body:
            body_bytes = json.dumps(body).encode()
            self.headers["Content-Length"] = str(len(body_bytes))
            self.rfile = MagicMock()
            self.rfile.read.return_value = body_bytes
        else:
            self.headers["Content-Length"] = "0"
            self.rfile = MagicMock()
            self.rfile.read.return_value = b"{}"


def _bypass_decorators():
    """Context manager patches to bypass require_permission, rate_limit, auth_rate_limit."""
    return (
        patch(
            "aragora.server.handlers.openclaw_gateway.require_permission",
            lambda *a, **kw: lambda f: f,
        ),
        patch(
            "aragora.server.handlers.openclaw_gateway.rate_limit",
            lambda *a, **kw: lambda f: f,
        ),
        patch(
            "aragora.server.handlers.openclaw_gateway.auth_rate_limit",
            lambda *a, **kw: lambda f: f,
        ),
    )


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def store():
    """Create a fresh store instance for each test."""
    return OpenClawGatewayStore()


@pytest.fixture
def handler(mock_server_context):
    """Create handler with mocked dependencies."""
    return OpenClawGatewayHandler(mock_server_context)


@pytest.fixture
def mock_user():
    """Create a standard mock user."""
    return MockUser()


@pytest.fixture
def admin_user():
    """Create a mock admin user."""
    return MockUser(
        user_id="admin-001",
        role="admin",
        permissions=["gateway:admin"],
    )


@pytest.fixture
def other_user():
    """Create a different mock user (for cross-ownership tests)."""
    return MockUser(
        user_id="user-002",
        email="other@example.com",
        org_id="org-002",
        role="user",
    )


def setup_handler_user(handler: OpenClawGatewayHandler, user: MockUser) -> None:
    """Configure handler to return the given user on authentication."""
    handler.get_current_user = MagicMock(return_value=user)


def call_with_bypassed_decorators(fn, *args, **kwargs):
    """Call a handler method with permission and rate-limit decorators bypassed."""
    p1, p2, p3 = _bypass_decorators()
    with p1, p2, p3:
        return fn(*args, **kwargs)


# ===========================================================================
# 1. Session Creation and Retrieval
# ===========================================================================


class TestSessionCreation:
    """Test session creation endpoint."""

    def test_create_session_returns_201(self, handler, mock_user, store):
        """Test that session creation returns HTTP 201."""
        setup_handler_user(handler, mock_user)
        req = MockRequestHandler(body={"config": {"timeout": 3600}})

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_create_session,
                {"config": {"timeout": 3600}},
                req,
            )

        assert result.status_code == 201

    def test_create_session_has_valid_id(self, handler, mock_user, store):
        """Test that created session has a UUID id."""
        setup_handler_user(handler, mock_user)
        req = MockRequestHandler(body={})

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_create_session, {}, req)

        body = json.loads(result.body)
        # Should be a valid UUID
        uuid.UUID(body["id"])

    def test_create_session_status_is_active(self, handler, mock_user, store):
        """Test that new session starts with active status."""
        setup_handler_user(handler, mock_user)
        req = MockRequestHandler(body={})

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_create_session, {}, req)

        body = json.loads(result.body)
        assert body["status"] == "active"

    def test_create_session_preserves_config(self, handler, mock_user, store):
        """Test that session creation preserves the config parameter."""
        setup_handler_user(handler, mock_user)
        config = {"timeout": 7200, "max_actions": 100}

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_create_session,
                {"config": config, "metadata": {}},
                MockRequestHandler(body={}),
            )

        body = json.loads(result.body)
        assert body["config"] == config

    def test_create_session_preserves_metadata(self, handler, mock_user, store):
        """Test that session creation preserves the metadata parameter."""
        setup_handler_user(handler, mock_user)
        metadata = {"source": "cli", "version": "1.0"}

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_create_session,
                {"metadata": metadata},
                MockRequestHandler(body={}),
            )

        body = json.loads(result.body)
        assert body["metadata"] == metadata

    def test_create_session_uses_current_user_id(self, handler, mock_user, store):
        """Test that session is created with the authenticated user's ID."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_create_session, {}, MockRequestHandler(body={})
            )

        body = json.loads(result.body)
        assert body["user_id"] == "user-001"

    def test_create_session_uses_current_tenant_id(self, handler, mock_user, store):
        """Test that session is created with the authenticated user's tenant/org."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_create_session, {}, MockRequestHandler(body={})
            )

        body = json.loads(result.body)
        assert body["tenant_id"] == "org-001"

    def test_create_session_generates_audit_entry(self, handler, mock_user, store):
        """Test that session creation logs an audit entry."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_create_session, {}, MockRequestHandler(body={})
            )

        entries, total = store.get_audit_log(action="session.create")
        assert total == 1
        assert entries[0].resource_type == "session"

    def test_create_session_timestamps_present(self, handler, mock_user, store):
        """Test that created session includes timestamp fields."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_create_session, {}, MockRequestHandler(body={})
            )

        body = json.loads(result.body)
        assert "created_at" in body
        assert "updated_at" in body
        assert "last_activity_at" in body


class TestSessionRetrieval:
    """Test session retrieval by ID."""

    def test_get_session_returns_200(self, handler, mock_user, store):
        """Test that getting an existing session returns 200."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_get_session, session.id, MockRequestHandler()
            )

        assert result.status_code == 200

    def test_get_session_returns_correct_data(self, handler, mock_user, store):
        """Test that session data matches what was created."""
        session = store.create_session(
            user_id="user-001",
            config={"key": "value"},
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_get_session, session.id, MockRequestHandler()
            )

        body = json.loads(result.body)
        assert body["id"] == session.id
        assert body["config"] == {"key": "value"}

    def test_get_session_not_found_returns_404(self, handler, mock_user, store):
        """Test that getting a non-existent session returns 404."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_get_session, "nonexistent-id", MockRequestHandler()
            )

        assert result.status_code == 404

    def test_get_session_access_denied_for_other_user(self, handler, other_user, store):
        """Test that a non-owner, non-admin user gets 403."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, other_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=False,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_get_session, session.id, MockRequestHandler()
                )

        assert result.status_code == 403

    def test_get_session_admin_can_access_other_user_session(self, handler, admin_user, store):
        """Test that an admin user can access another user's session."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, admin_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=True,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_get_session, session.id, MockRequestHandler()
                )

        assert result.status_code == 200


# ===========================================================================
# 2. Session Listing with Filters
# ===========================================================================


class TestSessionListing:
    """Test session listing endpoint."""

    def test_list_sessions_returns_200(self, handler, mock_user, store):
        """Test basic session listing returns 200."""
        store.create_session(user_id="user-001", tenant_id="org-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_sessions, {}, MockRequestHandler()
            )

        assert result.status_code == 200

    def test_list_sessions_includes_pagination_metadata(self, handler, mock_user, store):
        """Test that listing includes total, limit, offset fields."""
        store.create_session(user_id="user-001", tenant_id="org-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_sessions, {}, MockRequestHandler()
            )

        body = json.loads(result.body)
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_list_sessions_filter_by_active_status(self, handler, mock_user, store):
        """Test filtering sessions by active status."""
        s1 = store.create_session(user_id="user-001", tenant_id="org-001")
        store.create_session(user_id="user-001", tenant_id="org-001")
        store.update_session_status(s1.id, SessionStatus.CLOSED)
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_sessions,
                {"status": "active"},
                MockRequestHandler(),
            )

        body = json.loads(result.body)
        assert body["total"] == 1
        for s in body["sessions"]:
            assert s["status"] == "active"

    def test_list_sessions_filter_by_closed_status(self, handler, mock_user, store):
        """Test filtering sessions by closed status."""
        s1 = store.create_session(user_id="user-001", tenant_id="org-001")
        store.create_session(user_id="user-001", tenant_id="org-001")
        store.update_session_status(s1.id, SessionStatus.CLOSED)
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_sessions,
                {"status": "closed"},
                MockRequestHandler(),
            )

        body = json.loads(result.body)
        assert body["total"] == 1
        assert body["sessions"][0]["status"] == "closed"

    def test_list_sessions_invalid_status_returns_400(self, handler, mock_user, store):
        """Test that an invalid status filter returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_sessions,
                {"status": "nonexistent_status"},
                MockRequestHandler(),
            )

        assert result.status_code == 400

    def test_list_sessions_scoped_to_user(self, handler, mock_user, store):
        """Test that listing scopes sessions to the authenticated user."""
        store.create_session(user_id="user-001", tenant_id="org-001")
        store.create_session(user_id="other-user", tenant_id="org-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_sessions, {}, MockRequestHandler()
            )

        body = json.loads(result.body)
        assert body["total"] == 1

    def test_list_sessions_empty_result(self, handler, mock_user, store):
        """Test listing when no sessions exist."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_sessions, {}, MockRequestHandler()
            )

        body = json.loads(result.body)
        assert body["total"] == 0
        assert body["sessions"] == []


# ===========================================================================
# 3. Action Execution
# ===========================================================================


class TestActionExecution:
    """Test action execution endpoint."""

    def test_execute_action_returns_202(self, handler, mock_user, store):
        """Test successful action execution returns 202 Accepted."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_execute_action,
                {
                    "session_id": session.id,
                    "action_type": "browse",
                    "input": {"url": "https://example.com"},
                },
                MockRequestHandler(body={}),
            )

        assert result.status_code == 202

    def test_execute_action_returns_action_type(self, handler, mock_user, store):
        """Test action execution returns the correct action type."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_execute_action,
                {"session_id": session.id, "action_type": "click", "input": {}},
                MockRequestHandler(body={}),
            )

        body = json.loads(result.body)
        assert body["action_type"] == "click"

    def test_execute_action_status_is_running(self, handler, mock_user, store):
        """Test that action immediately transitions to running status."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_execute_action,
                {"session_id": session.id, "action_type": "browse", "input": {}},
                MockRequestHandler(body={}),
            )

        body = json.loads(result.body)
        assert body["status"] == "running"

    def test_execute_action_missing_session_id_returns_400(self, handler, mock_user, store):
        """Test that missing session_id returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_execute_action,
                {"action_type": "browse"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_execute_action_missing_action_type_returns_400(self, handler, mock_user, store):
        """Test that missing action_type returns 400."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_execute_action,
                {"session_id": session.id},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_execute_action_nonexistent_session_returns_404(self, handler, mock_user, store):
        """Test executing action on non-existent session returns 404."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_execute_action,
                {"session_id": "nonexistent", "action_type": "browse"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 404

    def test_execute_action_on_closed_session_returns_400(self, handler, mock_user, store):
        """Test executing action on a closed session returns 400."""
        session = store.create_session(user_id="user-001")
        store.update_session_status(session.id, SessionStatus.CLOSED)
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_execute_action,
                {"session_id": session.id, "action_type": "browse"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_execute_action_access_denied_for_other_user(self, handler, other_user, store):
        """Test executing action on another user's session returns 403."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, other_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=False,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_execute_action,
                    {"session_id": session.id, "action_type": "browse"},
                    MockRequestHandler(body={}),
                )

        assert result.status_code == 403

    def test_execute_action_admin_can_use_other_user_session(self, handler, admin_user, store):
        """Test that admin can execute actions on other user's sessions."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, admin_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=True,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_execute_action,
                    {"session_id": session.id, "action_type": "browse"},
                    MockRequestHandler(body={}),
                )

        assert result.status_code == 202

    def test_execute_action_creates_audit_entry(self, handler, mock_user, store):
        """Test that executing an action creates an audit entry."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_execute_action,
                {"session_id": session.id, "action_type": "browse", "input": {}},
                MockRequestHandler(body={}),
            )

        entries, total = store.get_audit_log(action="action.execute")
        assert total == 1


# ===========================================================================
# 4. Action Status and Cancellation
# ===========================================================================


class TestActionStatus:
    """Test action status retrieval."""

    def test_get_action_returns_200(self, handler, mock_user, store):
        """Test getting action status returns 200."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(session_id=session.id, action_type="browse", input_data={})
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_get_action, action.id, MockRequestHandler()
            )

        assert result.status_code == 200

    def test_get_action_not_found_returns_404(self, handler, mock_user, store):
        """Test getting non-existent action returns 404."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_get_action, "nonexistent", MockRequestHandler()
            )

        assert result.status_code == 404

    def test_get_action_access_denied_for_non_owner(self, handler, other_user, store):
        """Test that non-owner cannot see another user's action."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(session_id=session.id, action_type="browse", input_data={})
        setup_handler_user(handler, other_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=False,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_get_action, action.id, MockRequestHandler()
                )

        assert result.status_code == 403

    def test_get_action_includes_input_data(self, handler, mock_user, store):
        """Test that action response includes input_data."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(
            session_id=session.id,
            action_type="type",
            input_data={"text": "hello world"},
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_get_action, action.id, MockRequestHandler()
            )

        body = json.loads(result.body)
        assert body["input_data"] == {"text": "hello world"}


class TestActionCancellation:
    """Test action cancellation endpoint."""

    def test_cancel_pending_action_returns_200(self, handler, mock_user, store):
        """Test cancelling a pending action returns 200."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(session_id=session.id, action_type="browse", input_data={})
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_cancel_action, action.id, MockRequestHandler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["cancelled"] is True

    def test_cancel_running_action_returns_200(self, handler, mock_user, store):
        """Test cancelling a running action returns 200."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(session_id=session.id, action_type="browse", input_data={})
        store.update_action(action.id, status=ActionStatus.RUNNING)
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_cancel_action, action.id, MockRequestHandler()
            )

        assert result.status_code == 200

    def test_cancel_completed_action_returns_400(self, handler, mock_user, store):
        """Test cancelling a completed action returns 400."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(session_id=session.id, action_type="browse", input_data={})
        store.update_action(action.id, status=ActionStatus.COMPLETED)
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_cancel_action, action.id, MockRequestHandler()
            )

        assert result.status_code == 400

    def test_cancel_failed_action_returns_400(self, handler, mock_user, store):
        """Test cancelling a failed action returns 400."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(session_id=session.id, action_type="browse", input_data={})
        store.update_action(action.id, status=ActionStatus.FAILED)
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_cancel_action, action.id, MockRequestHandler()
            )

        assert result.status_code == 400

    def test_cancel_nonexistent_action_returns_404(self, handler, mock_user, store):
        """Test cancelling non-existent action returns 404."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_cancel_action, "nonexistent", MockRequestHandler()
            )

        assert result.status_code == 404

    def test_cancel_action_access_denied_for_non_owner(self, handler, other_user, store):
        """Test that non-owner cannot cancel another user's action."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(session_id=session.id, action_type="browse", input_data={})
        setup_handler_user(handler, other_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=False,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_cancel_action, action.id, MockRequestHandler()
                )

        assert result.status_code == 403

    def test_cancel_action_creates_audit_entry(self, handler, mock_user, store):
        """Test that cancelling an action creates an audit entry."""
        session = store.create_session(user_id="user-001")
        action = store.create_action(session_id=session.id, action_type="browse", input_data={})
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_cancel_action, action.id, MockRequestHandler()
            )

        entries, total = store.get_audit_log(action="action.cancel")
        assert total == 1


# ===========================================================================
# 5. Credential Storage (no secret in response)
# ===========================================================================


class TestCredentialStorage:
    """Test credential storage endpoint."""

    def test_store_credential_returns_201(self, handler, mock_user, store):
        """Test storing a credential returns 201."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {"name": "My Key", "type": "api_key", "secret": "secret123"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 201

    def test_store_credential_no_secret_in_response(self, handler, mock_user, store):
        """Test that secret value is never returned in the response."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {"name": "My Key", "type": "api_key", "secret": "super-secret-123"},
                MockRequestHandler(body={}),
            )

        body = json.loads(result.body)
        assert "secret" not in body
        # Also verify the secret value itself does not appear anywhere
        assert "super-secret-123" not in result.body.decode()

    def test_store_credential_returns_name_and_type(self, handler, mock_user, store):
        """Test that credential response includes name and type."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {"name": "OAuth Token", "type": "oauth_token", "secret": "tok123"},
                MockRequestHandler(body={}),
            )

        body = json.loads(result.body)
        assert body["name"] == "OAuth Token"
        assert body["credential_type"] == "oauth_token"

    def test_store_credential_missing_name_returns_400(self, handler, mock_user, store):
        """Test that missing name returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {"type": "api_key", "secret": "s"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_store_credential_missing_type_returns_400(self, handler, mock_user, store):
        """Test that missing type returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {"name": "Test", "secret": "s"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_store_credential_missing_secret_returns_400(self, handler, mock_user, store):
        """Test that missing secret returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {"name": "Test", "type": "api_key"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_store_credential_invalid_type_returns_400(self, handler, mock_user, store):
        """Test that invalid credential type returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {"name": "Test", "type": "banana", "secret": "s"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_store_credential_with_valid_expiry(self, handler, mock_user, store):
        """Test storing credential with a valid ISO 8601 expiry date."""
        expires = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {
                    "name": "Expiring Cred",
                    "type": "oauth_token",
                    "secret": "tok123",
                    "expires_at": expires,
                },
                MockRequestHandler(body={}),
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["expires_at"] is not None

    def test_store_credential_with_invalid_expiry_returns_400(self, handler, mock_user, store):
        """Test storing credential with invalid expiry format returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_store_credential,
                {
                    "name": "Bad Expiry",
                    "type": "api_key",
                    "secret": "s",
                    "expires_at": "not-a-date",
                },
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_store_credential_creates_audit_entry(self, handler, mock_user, store):
        """Test that storing a credential creates an audit entry."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_store_credential,
                {"name": "Key", "type": "api_key", "secret": "s"},
                MockRequestHandler(body={}),
            )

        entries, total = store.get_audit_log(action="credential.create")
        assert total == 1

    def test_store_all_credential_types(self, handler, mock_user, store):
        """Test that all valid credential types can be stored."""
        setup_handler_user(handler, mock_user)
        valid_types = [
            "api_key",
            "oauth_token",
            "password",
            "certificate",
            "ssh_key",
            "service_account",
        ]

        for ctype in valid_types:
            with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
                result = call_with_bypassed_decorators(
                    handler._handle_store_credential,
                    {"name": f"Cred-{ctype}", "type": ctype, "secret": "s"},
                    MockRequestHandler(body={}),
                )
            assert result.status_code == 201, f"Failed for credential type: {ctype}"


# ===========================================================================
# 6. Credential Listing and Deletion
# ===========================================================================


class TestCredentialListing:
    """Test credential listing endpoint."""

    def test_list_credentials_returns_200(self, handler, mock_user, store):
        """Test listing credentials returns 200."""
        store.store_credential(
            name="Test",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="user-001",
            tenant_id="org-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_credentials, {}, MockRequestHandler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "credentials" in body
        assert body["total"] >= 1

    def test_list_credentials_no_secret_values(self, handler, mock_user, store):
        """Test that listed credentials never contain secret values."""
        store.store_credential(
            name="Sensitive Key",
            credential_type=CredentialType.API_KEY,
            secret_value="VERY_SECRET_VALUE",
            user_id="user-001",
            tenant_id="org-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_credentials, {}, MockRequestHandler()
            )

        raw = result.body.decode()
        assert "VERY_SECRET_VALUE" not in raw
        body = json.loads(result.body)
        for cred in body["credentials"]:
            assert "secret" not in cred

    def test_list_credentials_filter_by_type(self, handler, mock_user, store):
        """Test listing credentials filtered by type."""
        store.store_credential(
            name="Key",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="user-001",
            tenant_id="org-001",
        )
        store.store_credential(
            name="Pass",
            credential_type=CredentialType.PASSWORD,
            secret_value="s",
            user_id="user-001",
            tenant_id="org-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_credentials,
                {"type": "api_key"},
                MockRequestHandler(),
            )

        body = json.loads(result.body)
        assert body["total"] == 1
        assert body["credentials"][0]["credential_type"] == "api_key"

    def test_list_credentials_invalid_type_returns_400(self, handler, mock_user, store):
        """Test listing with invalid credential type returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_credentials,
                {"type": "invalid_type"},
                MockRequestHandler(),
            )

        assert result.status_code == 400


class TestCredentialDeletion:
    """Test credential deletion endpoint."""

    def test_delete_credential_returns_200(self, handler, mock_user, store):
        """Test deleting a credential returns 200."""
        cred = store.store_credential(
            name="Delete Me",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="user-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_delete_credential, cred.id, MockRequestHandler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deleted"] is True

    def test_delete_credential_not_found_returns_404(self, handler, mock_user, store):
        """Test deleting non-existent credential returns 404."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_delete_credential, "nonexistent", MockRequestHandler()
            )

        assert result.status_code == 404

    def test_delete_credential_access_denied_for_non_owner(self, handler, other_user, store):
        """Test that non-owner cannot delete another user's credential."""
        cred = store.store_credential(
            name="Not Yours",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="user-001",
        )
        setup_handler_user(handler, other_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=False,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_delete_credential, cred.id, MockRequestHandler()
                )

        assert result.status_code == 403

    def test_delete_credential_creates_audit_entry(self, handler, mock_user, store):
        """Test that deleting a credential creates an audit entry."""
        cred = store.store_credential(
            name="Audit Me",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="user-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_delete_credential, cred.id, MockRequestHandler()
            )

        entries, total = store.get_audit_log(action="credential.delete")
        assert total == 1


# ===========================================================================
# 7. Credential Rotation
# ===========================================================================


class TestCredentialRotation:
    """Test credential rotation endpoint."""

    def test_rotate_credential_returns_200(self, handler, mock_user, store):
        """Test rotating a credential returns 200."""
        cred = store.store_credential(
            name="Rotate Me",
            credential_type=CredentialType.PASSWORD,
            secret_value="old_pass",
            user_id="user-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_rotate_credential,
                cred.id,
                {"secret": "new_pass"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["rotated"] is True

    def test_rotate_credential_updates_rotated_at(self, handler, mock_user, store):
        """Test that rotation updates the last_rotated_at timestamp."""
        cred = store.store_credential(
            name="Rotate Me",
            credential_type=CredentialType.PASSWORD,
            secret_value="old_pass",
            user_id="user-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_rotate_credential,
                cred.id,
                {"secret": "new_pass"},
                MockRequestHandler(body={}),
            )

        body = json.loads(result.body)
        assert body["rotated_at"] is not None

    def test_rotate_credential_not_found_returns_404(self, handler, mock_user, store):
        """Test rotating non-existent credential returns 404."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_rotate_credential,
                "nonexistent",
                {"secret": "new"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 404

    def test_rotate_credential_missing_secret_returns_400(self, handler, mock_user, store):
        """Test rotating without new secret returns 400."""
        cred = store.store_credential(
            name="Rotate Me",
            credential_type=CredentialType.PASSWORD,
            secret_value="old",
            user_id="user-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_rotate_credential,
                cred.id,
                {},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_rotate_credential_access_denied_for_non_owner(self, handler, other_user, store):
        """Test that non-owner cannot rotate another user's credential."""
        cred = store.store_credential(
            name="Not Yours",
            credential_type=CredentialType.PASSWORD,
            secret_value="old",
            user_id="user-001",
        )
        setup_handler_user(handler, other_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=False,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_rotate_credential,
                    cred.id,
                    {"secret": "new"},
                    MockRequestHandler(body={}),
                )

        assert result.status_code == 403

    def test_rotate_credential_creates_audit_entry(self, handler, mock_user, store):
        """Test that rotating a credential creates an audit entry."""
        cred = store.store_credential(
            name="Audit Me",
            credential_type=CredentialType.PASSWORD,
            secret_value="old",
            user_id="user-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_rotate_credential,
                cred.id,
                {"secret": "new"},
                MockRequestHandler(body={}),
            )

        entries, total = store.get_audit_log(action="credential.rotate")
        assert total == 1

    def test_rotate_credential_no_secret_in_response(self, handler, mock_user, store):
        """Test that the rotation response does not leak the new secret."""
        cred = store.store_credential(
            name="Rotate Me",
            credential_type=CredentialType.PASSWORD,
            secret_value="old",
            user_id="user-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_rotate_credential,
                cred.id,
                {"secret": "new_super_secret_value"},
                MockRequestHandler(body={}),
            )

        raw = result.body.decode()
        assert "new_super_secret_value" not in raw


# ===========================================================================
# 8. Health Endpoint
# ===========================================================================


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_returns_200_when_healthy(self, handler, store):
        """Test health endpoint returns 200 when everything is fine."""
        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = handler._handle_health(MockRequestHandler())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["healthy"] is True
        assert body["status"] == "healthy"

    def test_health_includes_timestamp(self, handler, store):
        """Test health response includes a timestamp."""
        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = handler._handle_health(MockRequestHandler())

        body = json.loads(result.body)
        assert "timestamp" in body

    def test_health_degraded_when_over_100_running(self, handler, store):
        """Test health status is degraded when running actions exceed 100."""
        session = store.create_session(user_id="user-001")
        for _ in range(101):
            action = store.create_action(session_id=session.id, action_type="browse", input_data={})
            store.update_action(action.id, status=ActionStatus.RUNNING)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = handler._handle_health(MockRequestHandler())

        body = json.loads(result.body)
        assert body["status"] == "degraded"

    def test_health_unhealthy_when_over_500_pending(self, handler, store):
        """Test health status is unhealthy when pending actions exceed 500."""
        session = store.create_session(user_id="user-001")
        for _ in range(501):
            store.create_action(session_id=session.id, action_type="browse", input_data={})

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = handler._handle_health(MockRequestHandler())

        body = json.loads(result.body)
        assert body["healthy"] is False
        assert body["status"] == "unhealthy"

    def test_health_returns_503_on_store_error(self, handler):
        """Test health returns 503 when store raises an exception."""
        mock_store = MagicMock()
        mock_store.get_metrics.side_effect = RuntimeError("Store unavailable")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = handler._handle_health(MockRequestHandler())

        assert result.status_code == 503
        body = json.loads(result.body)
        assert body["healthy"] is False
        assert body["status"] == "error"


# ===========================================================================
# 9. Metrics Endpoint
# ===========================================================================


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_metrics_returns_200(self, handler, mock_user, store):
        """Test metrics endpoint returns 200."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_metrics, MockRequestHandler())

        assert result.status_code == 200

    def test_metrics_includes_all_sections(self, handler, mock_user, store):
        """Test metrics response includes sessions, actions, credentials."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_metrics, MockRequestHandler())

        body = json.loads(result.body)
        assert "sessions" in body
        assert "actions" in body
        assert "credentials" in body
        assert "timestamp" in body

    def test_metrics_reflects_store_state(self, handler, mock_user, store):
        """Test that metrics accurately reflect the store state."""
        store.create_session(user_id="user-001")
        store.create_session(user_id="user-002")
        store.store_credential(
            name="Key",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="user-001",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_metrics, MockRequestHandler())

        body = json.loads(result.body)
        assert body["sessions"]["total"] == 2
        assert body["credentials"]["total"] == 1


# ===========================================================================
# 10. Audit Log with Filters
# ===========================================================================


class TestAuditLog:
    """Test audit log endpoint."""

    def test_audit_returns_200(self, handler, mock_user, store):
        """Test audit endpoint returns 200."""
        store.add_audit_entry(
            action="session.create",
            actor_id="user-001",
            resource_type="session",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_audit, {}, MockRequestHandler())

        assert result.status_code == 200

    def test_audit_includes_entries_and_total(self, handler, mock_user, store):
        """Test audit response includes entries array and total count."""
        store.add_audit_entry(
            action="session.create",
            actor_id="user-001",
            resource_type="session",
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_audit, {}, MockRequestHandler())

        body = json.loads(result.body)
        assert "entries" in body
        assert "total" in body
        assert body["total"] == 1

    def test_audit_filter_by_action(self, handler, mock_user, store):
        """Test filtering audit log by action type."""
        store.add_audit_entry(action="session.create", actor_id="u1", resource_type="session")
        store.add_audit_entry(action="credential.rotate", actor_id="u1", resource_type="credential")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_audit,
                {"action": "session.create"},
                MockRequestHandler(),
            )

        body = json.loads(result.body)
        assert body["total"] == 1
        assert body["entries"][0]["action"] == "session.create"

    def test_audit_filter_by_actor_id(self, handler, mock_user, store):
        """Test filtering audit log by actor ID."""
        store.add_audit_entry(action="session.create", actor_id="user-001", resource_type="session")
        store.add_audit_entry(
            action="session.create", actor_id="admin-001", resource_type="session"
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_audit,
                {"actor_id": "admin-001"},
                MockRequestHandler(),
            )

        body = json.loads(result.body)
        assert body["total"] == 1

    def test_audit_filter_by_resource_type(self, handler, mock_user, store):
        """Test filtering audit log by resource type."""
        store.add_audit_entry(action="session.create", actor_id="u1", resource_type="session")
        store.add_audit_entry(action="credential.create", actor_id="u1", resource_type="credential")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_audit,
                {"resource_type": "credential"},
                MockRequestHandler(),
            )

        body = json.loads(result.body)
        assert body["total"] == 1

    def test_audit_combined_filters(self, handler, mock_user, store):
        """Test audit log with multiple filters combined."""
        store.add_audit_entry(action="session.create", actor_id="user-001", resource_type="session")
        store.add_audit_entry(
            action="session.create", actor_id="admin-001", resource_type="session"
        )
        store.add_audit_entry(
            action="credential.create", actor_id="user-001", resource_type="credential"
        )
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_audit,
                {"action": "session.create", "actor_id": "user-001"},
                MockRequestHandler(),
            )

        body = json.loads(result.body)
        assert body["total"] == 1


# ===========================================================================
# 11. RBAC Permission Enforcement
# ===========================================================================


class TestRBACEnforcement:
    """Test RBAC permission enforcement through handler methods."""

    def test_user_id_extraction_authenticated(self, handler, mock_user):
        """Test user ID extraction for authenticated user."""
        setup_handler_user(handler, mock_user)
        result = handler._get_user_id(MockRequestHandler())
        assert result == "user-001"

    def test_user_id_extraction_anonymous(self, handler):
        """Test user ID extraction for anonymous user returns 'anonymous'."""
        handler.get_current_user = MagicMock(return_value=None)
        result = handler._get_user_id(MockRequestHandler())
        assert result == "anonymous"

    def test_tenant_id_extraction(self, handler, mock_user):
        """Test tenant ID extraction from user org_id."""
        setup_handler_user(handler, mock_user)
        result = handler._get_tenant_id(MockRequestHandler())
        assert result == "org-001"

    def test_tenant_id_extraction_no_org(self, handler):
        """Test tenant ID extraction when user has no org."""
        user = MockUser(org_id=None)
        setup_handler_user(handler, user)
        result = handler._get_tenant_id(MockRequestHandler())
        assert result is None

    def test_close_session_access_denied_for_non_owner(self, handler, other_user, store):
        """Test that non-owner cannot close another user's session."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, other_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=False,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_close_session, session.id, MockRequestHandler()
                )

        assert result.status_code == 403

    def test_end_session_access_denied_for_non_owner(self, handler, other_user, store):
        """Test that non-owner cannot end another user's session via POST."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, other_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.has_permission",
                return_value=False,
            ):
                result = call_with_bypassed_decorators(
                    handler._handle_end_session, session.id, MockRequestHandler()
                )

        assert result.status_code == 403


# ===========================================================================
# 12. Error Handling
# ===========================================================================


class TestErrorHandling:
    """Test error handling across handler methods."""

    def test_create_session_store_exception_returns_500(self, handler, mock_user):
        """Test session creation handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.create_session.side_effect = OSError("DB down")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(
                handler._handle_create_session, {}, MockRequestHandler(body={})
            )

        assert result.status_code == 500

    def test_execute_action_store_exception_returns_500(self, handler, mock_user):
        """Test action execution handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.get_session.side_effect = OSError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(
                handler._handle_execute_action,
                {"session_id": "x", "action_type": "browse"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 500

    def test_cancel_action_store_exception_returns_500(self, handler, mock_user):
        """Test action cancellation handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.get_action.side_effect = OSError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(
                handler._handle_cancel_action, "x", MockRequestHandler()
            )

        assert result.status_code == 500

    def test_get_session_store_exception_returns_500(self, handler, mock_user):
        """Test session retrieval handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.get_session.side_effect = OSError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(
                handler._handle_get_session, "x", MockRequestHandler()
            )

        assert result.status_code == 500

    def test_list_credentials_store_exception_returns_500(self, handler, mock_user):
        """Test credential listing handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.list_credentials.side_effect = RuntimeError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(
                handler._handle_list_credentials, {}, MockRequestHandler()
            )

        assert result.status_code == 500

    def test_delete_credential_store_exception_returns_500(self, handler, mock_user):
        """Test credential deletion handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.get_credential.side_effect = RuntimeError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(
                handler._handle_delete_credential, "x", MockRequestHandler()
            )

        assert result.status_code == 500

    def test_rotate_credential_store_exception_returns_500(self, handler, mock_user):
        """Test credential rotation handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.get_credential.side_effect = RuntimeError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(
                handler._handle_rotate_credential,
                "x",
                {"secret": "new"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 500

    def test_metrics_store_exception_returns_500(self, handler, mock_user):
        """Test metrics endpoint handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.get_metrics.side_effect = RuntimeError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(handler._handle_metrics, MockRequestHandler())

        assert result.status_code == 500

    def test_audit_store_exception_returns_500(self, handler, mock_user):
        """Test audit endpoint handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.get_audit_log.side_effect = RuntimeError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(handler._handle_audit, {}, MockRequestHandler())

        assert result.status_code == 500

    def test_close_session_store_exception_returns_500(self, handler, mock_user):
        """Test session close handles store exception."""
        setup_handler_user(handler, mock_user)
        mock_store = MagicMock()
        mock_store.get_session.side_effect = OSError("DB error")

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=mock_store):
            result = call_with_bypassed_decorators(
                handler._handle_close_session, "x", MockRequestHandler()
            )

        assert result.status_code == 500


# ===========================================================================
# Path Routing and Normalization
# ===========================================================================


class TestPathRouting:
    """Test request routing and path normalization."""

    def test_can_handle_base_openclaw_path(self, handler):
        """Test handler recognizes /api/gateway/openclaw/ prefix."""
        assert handler.can_handle("/api/gateway/openclaw/sessions")
        assert handler.can_handle("/api/gateway/openclaw/actions")
        assert handler.can_handle("/api/gateway/openclaw/credentials")
        assert handler.can_handle("/api/gateway/openclaw/health")

    def test_can_handle_v1_gateway_path(self, handler):
        """Test handler recognizes /api/v1/gateway/openclaw/ prefix."""
        assert handler.can_handle("/api/v1/gateway/openclaw/sessions")
        assert handler.can_handle("/api/v1/gateway/openclaw/actions/action-001")

    def test_can_handle_v1_openclaw_path(self, handler):
        """Test handler recognizes /api/v1/openclaw/ prefix."""
        assert handler.can_handle("/api/v1/openclaw/sessions")
        assert handler.can_handle("/api/v1/openclaw/credentials")

    def test_cannot_handle_unrelated_paths(self, handler):
        """Test handler rejects unrelated paths."""
        assert not handler.can_handle("/api/gateway/other/sessions")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/health")

    def test_normalize_v1_gateway_path(self, handler):
        """Test v1 gateway path normalization."""
        result = handler._normalize_path("/api/v1/gateway/openclaw/sessions")
        assert result == "/api/gateway/openclaw/sessions"

    def test_normalize_v1_openclaw_path(self, handler):
        """Test v1 openclaw path normalization."""
        result = handler._normalize_path("/api/v1/openclaw/sessions")
        assert result == "/api/gateway/openclaw/sessions"

    def test_normalize_base_path_unchanged(self, handler):
        """Test base path is not changed by normalization."""
        result = handler._normalize_path("/api/gateway/openclaw/actions")
        assert result == "/api/gateway/openclaw/actions"

    def test_get_handler_returns_none_for_unmatched(self, handler):
        """Test GET handler returns None for unmatched paths."""
        result = handler.handle("/api/gateway/openclaw/unrecognized", {}, MockRequestHandler())
        assert result is None

    def test_post_handler_returns_none_for_unmatched(self, handler):
        """Test POST handler returns None for unmatched paths."""
        result = handler.handle_post(
            "/api/gateway/openclaw/unrecognized", {}, MockRequestHandler(body={})
        )
        assert result is None

    def test_delete_handler_returns_none_for_unmatched(self, handler):
        """Test DELETE handler returns None for unmatched paths."""
        result = handler.handle_delete(
            "/api/gateway/openclaw/unrecognized", {}, MockRequestHandler()
        )
        assert result is None


# ===========================================================================
# Session Close and End Session
# ===========================================================================


class TestSessionClose:
    """Test session closing and ending."""

    def test_close_session_returns_200(self, handler, mock_user, store):
        """Test closing a session returns 200."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_close_session, session.id, MockRequestHandler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["closed"] is True

    def test_close_session_updates_status(self, handler, mock_user, store):
        """Test that closing a session updates its status to CLOSED."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_close_session, session.id, MockRequestHandler()
            )

        updated = store.get_session(session.id)
        assert updated.status == SessionStatus.CLOSED

    def test_close_session_not_found_returns_404(self, handler, mock_user, store):
        """Test closing non-existent session returns 404."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_close_session, "nonexistent", MockRequestHandler()
            )

        assert result.status_code == 404

    def test_end_session_returns_200(self, handler, mock_user, store):
        """Test ending a session via POST returns 200."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_end_session, session.id, MockRequestHandler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True

    def test_end_session_not_found_returns_404(self, handler, mock_user, store):
        """Test ending non-existent session via POST returns 404."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_end_session, "nonexistent", MockRequestHandler()
            )

        assert result.status_code == 404

    def test_close_session_creates_audit_entry(self, handler, mock_user, store):
        """Test that closing a session creates an audit entry."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_close_session, session.id, MockRequestHandler()
            )

        entries, total = store.get_audit_log(action="session.close")
        assert total == 1

    def test_end_session_creates_audit_entry(self, handler, mock_user, store):
        """Test that ending a session creates an audit entry."""
        session = store.create_session(user_id="user-001")
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_end_session, session.id, MockRequestHandler()
            )

        entries, total = store.get_audit_log(action="session.end")
        assert total == 1


# ===========================================================================
# Stats Endpoint
# ===========================================================================


class TestStatsEndpoint:
    """Test stats endpoint."""

    def test_stats_returns_200(self, handler, mock_user, store):
        """Test stats endpoint returns 200."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_stats, MockRequestHandler())

        assert result.status_code == 200

    def test_stats_includes_expected_fields(self, handler, mock_user, store):
        """Test stats response includes expected fields."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(handler._handle_stats, MockRequestHandler())

        body = json.loads(result.body)
        assert "active_sessions" in body
        assert "pending_approvals" in body
        assert "timestamp" in body


# ===========================================================================
# Handler Factory and Singleton
# ===========================================================================


class TestHandlerFactory:
    """Test handler factory and store singleton."""

    def test_factory_returns_handler_instance(self, mock_server_context):
        """Test factory function returns OpenClawGatewayHandler instance."""
        h = get_openclaw_gateway_handler(mock_server_context)
        assert isinstance(h, OpenClawGatewayHandler)

    def test_store_singleton_returns_same_instance(self):
        """Test that _get_store returns the same instance on repeated calls."""
        import aragora.server.handlers.openclaw_gateway as module

        module._store = None
        s1 = _get_store()
        s2 = _get_store()
        assert s1 is s2
        module._store = None  # Reset for other tests


# ===========================================================================
# Data Model Tests
# ===========================================================================


class TestDataModels:
    """Test data model serialization edge cases."""

    def test_session_to_dict_none_tenant(self):
        """Test session serialization with None tenant_id."""
        now = datetime.now(timezone.utc)
        session = Session(
            id="s1",
            user_id="u1",
            tenant_id=None,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            last_activity_at=now,
        )
        d = session.to_dict()
        assert d["tenant_id"] is None

    def test_action_to_dict_null_optional_fields(self):
        """Test action serialization with null optional fields."""
        now = datetime.now(timezone.utc)
        action = Action(
            id="a1",
            session_id="s1",
            action_type="browse",
            status=ActionStatus.PENDING,
            input_data={},
            output_data=None,
            error=None,
            created_at=now,
            started_at=None,
            completed_at=None,
        )
        d = action.to_dict()
        assert d["output_data"] is None
        assert d["error"] is None
        assert d["started_at"] is None
        assert d["completed_at"] is None

    def test_credential_to_dict_excludes_secret(self):
        """Test credential serialization never includes 'secret' key."""
        now = datetime.now(timezone.utc)
        cred = Credential(
            id="c1",
            name="Key",
            credential_type=CredentialType.API_KEY,
            user_id="u1",
            tenant_id=None,
            created_at=now,
            updated_at=now,
            last_rotated_at=None,
            expires_at=None,
        )
        d = cred.to_dict()
        assert "secret" not in d
        assert "secret_value" not in d

    def test_audit_entry_to_dict_with_details(self):
        """Test audit entry serialization with details."""
        now = datetime.now(timezone.utc)
        entry = AuditEntry(
            id="ae1",
            timestamp=now,
            action="test.action",
            actor_id="u1",
            resource_type="test",
            resource_id="r1",
            result="success",
            details={"ip": "127.0.0.1", "method": "POST"},
        )
        d = entry.to_dict()
        assert d["details"]["ip"] == "127.0.0.1"
        assert d["result"] == "success"

    def test_session_default_config_and_metadata(self):
        """Test session defaults for config and metadata."""
        now = datetime.now(timezone.utc)
        session = Session(
            id="s1",
            user_id="u1",
            tenant_id=None,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            last_activity_at=now,
        )
        assert session.config == {}
        assert session.metadata == {}

    def test_action_default_metadata(self):
        """Test action defaults for metadata."""
        now = datetime.now(timezone.utc)
        action = Action(
            id="a1",
            session_id="s1",
            action_type="browse",
            status=ActionStatus.PENDING,
            input_data={},
            output_data=None,
            error=None,
            created_at=now,
            started_at=None,
            completed_at=None,
        )
        assert action.metadata == {}

    def test_all_session_statuses(self):
        """Test all session status enum values."""
        expected = {"active", "idle", "closing", "closed", "error"}
        actual = {s.value for s in SessionStatus}
        assert actual == expected

    def test_all_action_statuses(self):
        """Test all action status enum values."""
        expected = {"pending", "running", "completed", "failed", "cancelled", "timeout"}
        actual = {s.value for s in ActionStatus}
        assert actual == expected

    def test_all_credential_types(self):
        """Test all credential type enum values."""
        expected = {
            "api_key",
            "oauth_token",
            "password",
            "certificate",
            "ssh_key",
            "service_account",
        }
        actual = {t.value for t in CredentialType}
        assert actual == expected


# ===========================================================================
# Store Behavior Tests
# ===========================================================================


class TestStoreBehavior:
    """Test store behaviors not covered by handler tests."""

    def test_store_credential_stores_secret_separately(self, store):
        """Test that secret value is stored in separate internal dict."""
        cred = store.store_credential(
            name="Key",
            credential_type=CredentialType.API_KEY,
            secret_value="my-secret",
            user_id="u1",
        )
        assert cred.id in store._credential_secrets
        assert store._credential_secrets[cred.id] == "my-secret"

    def test_delete_credential_removes_secret(self, store):
        """Test that deleting a credential also removes its secret."""
        cred = store.store_credential(
            name="Key",
            credential_type=CredentialType.API_KEY,
            secret_value="my-secret",
            user_id="u1",
        )
        store.delete_credential(cred.id)
        assert cred.id not in store._credential_secrets

    def test_rotate_credential_updates_secret(self, store):
        """Test that rotation actually changes the stored secret."""
        cred = store.store_credential(
            name="Key",
            credential_type=CredentialType.API_KEY,
            secret_value="old-secret",
            user_id="u1",
        )
        store.rotate_credential(cred.id, "new-secret")
        assert store._credential_secrets[cred.id] == "new-secret"

    def test_audit_log_truncation_at_10000(self, store):
        """Test that audit log is truncated to 10000 entries."""
        for i in range(10005):
            store.add_audit_entry(action="test", actor_id="u1", resource_type="test")
        assert len(store._audit_log) <= 10000

    def test_list_sessions_sorted_by_created_at_descending(self, store):
        """Test that sessions are returned sorted by created_at descending."""
        import time

        s1 = store.create_session(user_id="u1")
        time.sleep(0.01)
        s2 = store.create_session(user_id="u1")

        sessions, _ = store.list_sessions()
        assert sessions[0].id == s2.id  # Most recent first

    def test_list_credentials_sorted_by_created_at_descending(self, store):
        """Test that credentials are sorted by created_at descending."""
        import time

        c1 = store.store_credential(
            name="C1",
            credential_type=CredentialType.API_KEY,
            secret_value="s1",
            user_id="u1",
        )
        time.sleep(0.01)
        c2 = store.store_credential(
            name="C2",
            credential_type=CredentialType.API_KEY,
            secret_value="s2",
            user_id="u1",
        )

        creds, _ = store.list_credentials()
        assert creds[0].id == c2.id  # Most recent first

    def test_metrics_by_status_breakdown(self, store):
        """Test that metrics include per-status breakdowns."""
        s1 = store.create_session(user_id="u1")
        s2 = store.create_session(user_id="u2")
        store.update_session_status(s2.id, SessionStatus.IDLE)

        action1 = store.create_action(session_id=s1.id, action_type="browse", input_data={})
        store.update_action(action1.id, status=ActionStatus.RUNNING)
        store.create_action(session_id=s1.id, action_type="click", input_data={})

        metrics = store.get_metrics()
        assert metrics["sessions"]["by_status"]["active"] == 1
        assert metrics["sessions"]["by_status"]["idle"] == 1
        assert metrics["actions"]["by_status"]["pending"] == 1
        assert metrics["actions"]["by_status"]["running"] == 1

    def test_credentials_by_type_breakdown(self, store):
        """Test that metrics include per-type credential breakdowns."""
        store.store_credential(
            name="K1",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="u1",
        )
        store.store_credential(
            name="K2",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="u1",
        )
        store.store_credential(
            name="P1",
            credential_type=CredentialType.PASSWORD,
            secret_value="s",
            user_id="u1",
        )

        metrics = store.get_metrics()
        assert metrics["credentials"]["by_type"]["api_key"] == 2
        assert metrics["credentials"]["by_type"]["password"] == 1

    def test_update_action_running_does_not_overwrite_started_at(self, store):
        """Test that setting RUNNING again does not overwrite started_at."""
        action = store.create_action(session_id="s1", action_type="browse", input_data={})
        store.update_action(action.id, status=ActionStatus.RUNNING)
        original_started = action.started_at

        # Setting RUNNING again should not change started_at
        store.update_action(action.id, status=ActionStatus.RUNNING)
        assert action.started_at == original_started

    def test_update_action_cancelled_sets_completed_at(self, store):
        """Test that cancelling an action sets completed_at."""
        action = store.create_action(session_id="s1", action_type="browse", input_data={})
        store.update_action(action.id, status=ActionStatus.CANCELLED)
        assert action.completed_at is not None


# ===========================================================================
# Policy and Approvals Endpoints
# ===========================================================================


class TestPolicyEndpoints:
    """Test policy rule and approval endpoints."""

    def test_get_policy_rules_returns_200(self, handler, mock_user, store):
        """Test getting policy rules returns 200."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_get_policy_rules, {}, MockRequestHandler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "rules" in body
        assert "total" in body

    def test_add_policy_rule_returns_201(self, handler, mock_user, store):
        """Test adding a policy rule returns 201."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_add_policy_rule,
                {"name": "deny-delete", "action_types": ["delete"], "decision": "deny"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 201

    def test_add_policy_rule_missing_name_returns_400(self, handler, mock_user, store):
        """Test adding policy rule without name returns 400."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_add_policy_rule,
                {"action_types": ["delete"]},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 400

    def test_list_approvals_returns_200(self, handler, mock_user, store):
        """Test listing approvals returns 200."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_list_approvals, {}, MockRequestHandler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "approvals" in body

    def test_approve_action_returns_200(self, handler, mock_user, store):
        """Test approving an action returns 200."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_approve_action,
                "approval-001",
                {"reason": "Looks good"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True

    def test_deny_action_returns_200(self, handler, mock_user, store):
        """Test denying an action returns 200."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            result = call_with_bypassed_decorators(
                handler._handle_deny_action,
                "approval-001",
                {"reason": "Not allowed"},
                MockRequestHandler(body={}),
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True

    def test_approve_action_creates_audit_entry(self, handler, mock_user, store):
        """Test that approving an action creates an audit entry."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_approve_action,
                "approval-001",
                {"reason": "LGTM"},
                MockRequestHandler(body={}),
            )

        entries, total = store.get_audit_log(action="approval.approve")
        assert total == 1

    def test_deny_action_creates_audit_entry(self, handler, mock_user, store):
        """Test that denying an action creates an audit entry."""
        setup_handler_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=store):
            call_with_bypassed_decorators(
                handler._handle_deny_action,
                "approval-001",
                {"reason": "Nope"},
                MockRequestHandler(body={}),
            )

        entries, total = store.get_audit_log(action="approval.deny")
        assert total == 1
