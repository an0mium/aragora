"""Tests for workspace handler (aragora/server/handlers/workspace_module.py).

Covers all routes and behavior of the WorkspaceHandler class:
- can_handle() routing for all ROUTES
- Workspace CRUD: POST/GET/DELETE /api/v1/workspaces
- Member management: POST/DELETE /api/v1/workspaces/{id}/members
- Retention policies: CRUD + execute + expiring
- Classification: POST /api/v1/classify, GET /api/v1/classify/policy/{level}
- Audit log: entries, report, verify, actor history, resource history, denied
- Invite management: create, list, cancel, resend, accept
- RBAC profiles and workspace roles
- Rate limiting, input validation, error handling, cross-tenant checks
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.workspace_module import (
    WorkspaceHandler,
    _invalidate_audit_cache,
    _invalidate_permission_cache,
    _invalidate_retention_cache,
    _retention_policy_cache,
    _permission_cache,
    _audit_query_cache,
    get_workspace_cache_stats,
)
from aragora.server.handlers.utils.rate_limit import _limiters, clear_all_limiters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for WorkspaceHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        path: str = "/",
    ):
        self.command = method
        self.path = path
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock data classes
# ---------------------------------------------------------------------------


@dataclass
class MockWorkspace:
    id: str = "ws-001"
    name: str = "Test Workspace"
    organization_id: str = "org-001"
    created_by: str = "user-001"
    members: list[str] = field(default_factory=list)
    rbac_profile: str = "lite"
    member_roles: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "organization_id": self.organization_id,
            "created_by": self.created_by,
            "members": self.members,
            "rbac_profile": self.rbac_profile,
            "member_roles": self.member_roles,
        }


@dataclass
class MockRetentionPolicy:
    id: str = "pol-001"
    name: str = "Test Policy"
    description: str = "Test retention policy"
    retention_days: int = 90
    action: MagicMock = field(default_factory=lambda: MagicMock(value="delete"))
    enabled: bool = True
    applies_to: list[str] = field(default_factory=lambda: ["documents"])
    workspace_ids: list[str] | None = None
    grace_period_days: int = 7
    notify_before_days: int = 3
    exclude_sensitivity_levels: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: datetime | None = None


@dataclass
class MockExecutionReport:
    items_deleted: int = 5
    items_evaluated: int = 20

    def to_dict(self) -> dict:
        return {
            "items_deleted": self.items_deleted,
            "items_evaluated": self.items_evaluated,
        }


@dataclass
class MockClassificationResult:
    level: MagicMock = field(default_factory=lambda: MagicMock(value="confidential"))
    confidence: float = 0.95

    def to_dict(self) -> dict:
        return {"level": self.level.value, "confidence": self.confidence}


@dataclass
class MockAuditEntry:
    id: str = "entry-001"
    action: str = "create"

    def to_dict(self) -> dict:
        return {"id": self.id, "action": self.action}


# ---------------------------------------------------------------------------
# Auth context mock
# ---------------------------------------------------------------------------


def _make_auth_ctx(
    authenticated: bool = True,
    user_id: str = "user-001",
    org_id: str = "org-001",
    email: str = "user@test.com",
    role: str = "admin",
):
    ctx = MagicMock()
    ctx.authenticated = authenticated
    ctx.is_authenticated = authenticated
    ctx.user_id = user_id
    ctx.org_id = org_id
    ctx.email = email
    ctx.role = role
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a WorkspaceHandler with an empty server context."""
    return WorkspaceHandler(server_context={})


@pytest.fixture
def http_handler():
    """Create a default MockHTTPHandler."""
    return MockHTTPHandler()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset all rate limiters before each test."""
    clear_all_limiters()
    for name, limiter in _limiters.items():
        limiter._buckets = defaultdict(list)
    yield
    clear_all_limiters()


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear all module-level caches before each test."""
    _retention_policy_cache.clear()
    _permission_cache.clear()
    _audit_query_cache.clear()
    yield
    _retention_policy_cache.clear()
    _permission_cache.clear()
    _audit_query_cache.clear()


@pytest.fixture
def mock_auth_ctx():
    return _make_auth_ctx()


@pytest.fixture
def mock_isolation_manager():
    mgr = MagicMock()
    mgr.create_workspace = AsyncMock(return_value=MockWorkspace())
    mgr.list_workspaces = AsyncMock(return_value=[MockWorkspace()])
    mgr.get_workspace = AsyncMock(return_value=MockWorkspace())
    mgr.delete_workspace = AsyncMock(return_value=None)
    mgr.add_member = AsyncMock(return_value=None)
    mgr.remove_member = AsyncMock(return_value=None)
    mgr.list_members = AsyncMock(return_value=[])
    return mgr


@pytest.fixture
def mock_retention_manager():
    mgr = MagicMock()
    mgr.list_policies = MagicMock(return_value=[MockRetentionPolicy()])
    mgr.get_policy = MagicMock(return_value=MockRetentionPolicy())
    mgr.create_policy = MagicMock(return_value=MockRetentionPolicy())
    mgr.update_policy = MagicMock(return_value=MockRetentionPolicy())
    mgr.delete_policy = MagicMock(return_value=None)
    mgr.execute_policy = AsyncMock(return_value=MockExecutionReport())
    mgr.check_expiring_soon = AsyncMock(return_value=[])
    return mgr


@pytest.fixture
def mock_audit_log():
    audit = MagicMock()
    audit.log = AsyncMock(return_value=None)
    audit.query = AsyncMock(return_value=[MockAuditEntry()])
    audit.generate_compliance_report = AsyncMock(
        return_value={"report_id": "rpt-001", "status": "complete"}
    )
    audit.verify_integrity = AsyncMock(return_value=(True, []))
    audit.get_actor_history = AsyncMock(return_value=[MockAuditEntry()])
    audit.get_resource_history = AsyncMock(return_value=[MockAuditEntry()])
    audit.get_denied_access_attempts = AsyncMock(return_value=[MockAuditEntry()])
    return audit


@pytest.fixture
def mock_classifier():
    cls = MagicMock()
    cls.classify = AsyncMock(return_value=MockClassificationResult())
    cls.get_level_policy = MagicMock(return_value={"encryption": True, "access_control": "strict"})
    return cls


def _patch_handler(
    handler,
    mock_isolation_manager,
    mock_retention_manager,
    mock_audit_log,
    mock_classifier,
    auth_ctx=None,
):
    """Patch handler internals with mocks."""
    if auth_ctx is None:
        auth_ctx = _make_auth_ctx()

    handler._get_isolation_manager = MagicMock(return_value=mock_isolation_manager)
    handler._get_retention_manager = MagicMock(return_value=mock_retention_manager)
    handler._get_audit_log = MagicMock(return_value=mock_audit_log)
    handler._get_classifier = MagicMock(return_value=mock_classifier)
    handler._get_user_store = MagicMock(return_value=None)

    def _sync_run_async(coro):
        if not hasattr(coro, "__await__"):
            return coro
        try:
            coro.__await__().__next__()
        except StopIteration as e:
            return e.value
        raise RuntimeError("coroutine did not return")

    handler._run_async = _sync_run_async
    handler._check_rbac_permission = MagicMock(return_value=None)
    return handler


# ---------------------------------------------------------------------------
# can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test can_handle route matching."""

    def test_handles_workspaces(self, handler):
        assert handler.can_handle("/api/v1/workspaces") is True

    def test_handles_workspaces_profiles(self, handler):
        assert handler.can_handle("/api/v1/workspaces/profiles") is True

    def test_handles_workspaces_with_id(self, handler):
        assert handler.can_handle("/api/v1/workspaces/ws-001") is True

    def test_handles_retention_policies(self, handler):
        assert handler.can_handle("/api/v1/retention/policies") is True

    def test_handles_retention_expiring(self, handler):
        assert handler.can_handle("/api/v1/retention/expiring") is True

    def test_handles_classify(self, handler):
        assert handler.can_handle("/api/v1/classify") is True

    def test_handles_audit_entries(self, handler):
        assert handler.can_handle("/api/v1/audit/entries") is True

    def test_handles_audit_report(self, handler):
        assert handler.can_handle("/api/v1/audit/report") is True

    def test_handles_audit_verify(self, handler):
        assert handler.can_handle("/api/v1/audit/verify") is True

    def test_handles_audit_actor(self, handler):
        assert handler.can_handle("/api/v1/audit/actor") is True

    def test_handles_audit_resource(self, handler):
        assert handler.can_handle("/api/v1/audit/resource") is True

    def test_handles_audit_denied(self, handler):
        assert handler.can_handle("/api/v1/audit/denied") is True

    def test_handles_invites(self, handler):
        assert handler.can_handle("/api/v1/invites") is True

    def test_does_not_handle_unknown(self, handler):
        assert handler.can_handle("/api/v1/unknown") is False

    def test_does_not_handle_billing(self, handler):
        assert handler.can_handle("/api/v1/billing") is False


# ---------------------------------------------------------------------------
# Workspace CRUD tests
# ---------------------------------------------------------------------------


class TestCreateWorkspace:
    """Test POST /api/v1/workspaces."""

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_create_workspace_success(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"name": "My Workspace"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces", {}, http, method="POST")

        assert _status(result) == 201
        body = _body(result)
        assert body["message"] == "Workspace created successfully"
        assert "workspace" in body

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_create_workspace_missing_name(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces", {}, http, method="POST")

        assert _status(result) == 400
        assert "name is required" in _body(result).get("error", "")

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_create_workspace_cross_tenant_rejected(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(
            body={"name": "Evil WS", "organization_id": "other-org"}, method="POST"
        )

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(org_id="org-001"),
        ):
            result = handler.handle("/api/v1/workspaces", {}, http, method="POST")

        assert _status(result) == 403
        assert "another organization" in _body(result).get("error", "")

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_create_workspace_unauthenticated(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"name": "WS"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(authenticated=False),
        ):
            result = handler.handle("/api/v1/workspaces", {}, http, method="POST")

        assert _status(result) == 401

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_create_workspace_no_org_id(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"name": "WS"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(org_id=None),
        ):
            result = handler.handle("/api/v1/workspaces", {}, http, method="POST")

        assert _status(result) == 400
        assert "organization_id" in _body(result).get("error", "")


class TestListWorkspaces:
    """Test GET /api/v1/workspaces."""

    def test_list_workspaces_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "workspaces" in body
        assert body["total"] == 1

    def test_list_workspaces_cross_tenant_rejected(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(org_id="org-001"),
        ):
            result = handler.handle(
                "/api/v1/workspaces",
                {"organization_id": "other-org"},
                http,
                method="GET",
            )

        assert _status(result) == 403


class TestGetWorkspace:
    """Test GET /api/v1/workspaces/{id}."""

    def test_get_workspace_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "workspace" in body

    def test_get_workspace_access_denied(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        from aragora.privacy import AccessDeniedException

        mock_isolation_manager.get_workspace = AsyncMock(
            side_effect=AccessDeniedException(
                "denied", workspace_id="ws-001", actor="user-001", action="access"
            )
        )
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001", {}, http, method="GET")

        assert _status(result) == 403


class TestDeleteWorkspace:
    """Test DELETE /api/v1/workspaces/{id}."""

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_delete_workspace_success(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001", {}, http, method="DELETE")

        assert _status(result) == 200
        assert "deleted" in _body(result).get("message", "").lower()

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_delete_workspace_access_denied(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        from aragora.privacy import AccessDeniedException

        mock_isolation_manager.delete_workspace = AsyncMock(
            side_effect=AccessDeniedException(
                "denied", workspace_id="ws-001", actor="user-001", action="access"
            )
        )
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001", {}, http, method="DELETE")

        assert _status(result) == 403


# ---------------------------------------------------------------------------
# Workspace ID validation
# ---------------------------------------------------------------------------


class TestWorkspaceIdValidation:
    """Test workspace ID validation in routes."""

    def test_invalid_workspace_id_get(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            # Path traversal attempt -- single segment so it reaches ID validation
            result = handler.handle("/api/v1/workspaces/..", {}, http, method="GET")

        assert _status(result) == 400


# ---------------------------------------------------------------------------
# Member management tests
# ---------------------------------------------------------------------------


class TestAddMember:
    """Test POST /api/v1/workspaces/{id}/members."""

    @patch("aragora.server.handlers.workspace.members.emit_handler_event")
    def test_add_member_success(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"user_id": "user-002"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001/members", {}, http, method="POST")

        assert _status(result) == 201

    @patch("aragora.server.handlers.workspace.members.emit_handler_event")
    def test_add_member_missing_user_id(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001/members", {}, http, method="POST")

        assert _status(result) == 400
        assert "user_id" in _body(result).get("error", "")


class TestRemoveMember:
    """Test DELETE /api/v1/workspaces/{id}/members/{user_id}."""

    def test_remove_member_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/members/user-002",
                {},
                http,
                method="DELETE",
            )

        assert _status(result) == 200

    def test_remove_member_access_denied(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        from aragora.privacy import AccessDeniedException

        mock_isolation_manager.remove_member = AsyncMock(
            side_effect=AccessDeniedException(
                "denied", workspace_id="ws-001", actor="user-001", action="access"
            )
        )
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/members/user-002",
                {},
                http,
                method="DELETE",
            )

        assert _status(result) == 403


# ---------------------------------------------------------------------------
# Retention policy tests
# ---------------------------------------------------------------------------


class TestListPolicies:
    """Test GET /api/v1/retention/policies."""

    def test_list_policies_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/policies", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "policies" in body
        assert body["total"] == 1

    def test_list_policies_with_workspace_filter(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/retention/policies",
                {"workspace_id": "ws-001"},
                http,
                method="GET",
            )

        assert _status(result) == 200


class TestCreatePolicy:
    """Test POST /api/v1/retention/policies."""

    def test_create_policy_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"name": "Test Policy"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/policies", {}, http, method="POST")

        assert _status(result) == 201
        assert "Policy created" in _body(result).get("message", "")

    def test_create_policy_missing_name(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/policies", {}, http, method="POST")

        assert _status(result) == 400
        assert "name is required" in _body(result).get("error", "")

    def test_create_policy_invalid_action(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(
            body={"name": "Bad Policy", "action": "invalid_action"}, method="POST"
        )

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/policies", {}, http, method="POST")

        assert _status(result) == 400
        assert "Invalid action" in _body(result).get("error", "")


class TestGetPolicy:
    """Test GET /api/v1/retention/policies/{id}."""

    def test_get_policy_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/policies/pol-001", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "policy" in body

    def test_get_policy_not_found(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        mock_retention_manager.get_policy.return_value = None
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/retention/policies/pol-missing", {}, http, method="GET"
            )

        assert _status(result) == 404


class TestUpdatePolicy:
    """Test PUT /api/v1/retention/policies/{id}."""

    def test_update_policy_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"name": "Updated"}, method="PUT")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/policies/pol-001", {}, http, method="PUT")

        assert _status(result) == 200
        assert "updated" in _body(result).get("message", "").lower()

    def test_update_policy_not_found(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        mock_retention_manager.update_policy.side_effect = ValueError("Not found")
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"name": "Updated"}, method="PUT")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/retention/policies/pol-missing", {}, http, method="PUT"
            )

        assert _status(result) == 404

    def test_update_policy_invalid_action(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"action": "bad_action"}, method="PUT")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/policies/pol-001", {}, http, method="PUT")

        assert _status(result) == 400


class TestDeletePolicy:
    """Test DELETE /api/v1/retention/policies/{id}."""

    def test_delete_policy_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/policies/pol-001", {}, http, method="DELETE")

        assert _status(result) == 200
        assert "deleted" in _body(result).get("message", "").lower()


class TestExecutePolicy:
    """Test POST /api/v1/retention/policies/{id}/execute."""

    def test_execute_policy_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/retention/policies/pol-001/execute",
                {},
                http,
                method="POST",
            )

        assert _status(result) == 200
        body = _body(result)
        assert "report" in body
        assert body["dry_run"] is False

    def test_execute_policy_dry_run(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/retention/policies/pol-001/execute",
                {"dry_run": "true"},
                http,
                method="POST",
            )

        assert _status(result) == 200
        assert _body(result)["dry_run"] is True

    def test_execute_policy_not_found(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        mock_retention_manager.execute_policy = AsyncMock(side_effect=ValueError("Not found"))
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/retention/policies/pol-missing/execute",
                {},
                http,
                method="POST",
            )

        assert _status(result) == 404


class TestExpiringItems:
    """Test GET /api/v1/retention/expiring."""

    def test_expiring_items_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/expiring", {"days": "7"}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "expiring" in body
        assert body["days_ahead"] == 7

    def test_expiring_items_default_days(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/expiring", {}, http, method="GET")

        assert _status(result) == 200
        assert _body(result)["days_ahead"] == 14


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassifyContent:
    """Test POST /api/v1/classify."""

    @patch("aragora.server.handlers.workspace.settings.emit_handler_event")
    def test_classify_success(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(
            body={"content": "Sensitive data here", "document_id": "doc-001"},
            method="POST",
        )

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/classify", {}, http, method="POST")

        assert _status(result) == 200
        body = _body(result)
        assert "classification" in body

    @patch("aragora.server.handlers.workspace.settings.emit_handler_event")
    def test_classify_missing_content(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/classify", {}, http, method="POST")

        assert _status(result) == 400
        assert "content is required" in _body(result).get("error", "")


class TestGetLevelPolicy:
    """Test GET /api/v1/classify/policy/{level}."""

    def test_get_level_policy_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        # Patch SensitivityLevel to accept our test value
        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch(
                "aragora.server.handlers.workspace_module.SensitivityLevel",
                side_effect=lambda x: MagicMock(value=x),
            ),
        ):
            result = handler.handle("/api/v1/classify/policy/confidential", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "policy" in body

    def test_get_level_policy_invalid_level(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch(
                "aragora.server.handlers.workspace_module.SensitivityLevel",
                side_effect=ValueError("invalid"),
            ),
        ):
            result = handler.handle("/api/v1/classify/policy/invalid_level", {}, http, method="GET")

        assert _status(result) == 400
        assert "Invalid level" in _body(result).get("error", "")


# ---------------------------------------------------------------------------
# Audit tests
# ---------------------------------------------------------------------------


class TestQueryAudit:
    """Test GET /api/v1/audit/entries."""

    def test_query_audit_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/audit/entries", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "entries" in body
        assert body["total"] == 1

    def test_query_audit_with_date_filters(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/audit/entries",
                {"start_date": "2025-01-01T00:00:00", "end_date": "2025-12-31T23:59:59"},
                http,
                method="GET",
            )

        assert _status(result) == 200

    def test_query_audit_invalid_date(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/audit/entries",
                {"start_date": "not-a-date"},
                http,
                method="GET",
            )

        assert _status(result) == 400
        assert "date format" in _body(result).get("error", "").lower()


class TestAuditReport:
    """Test GET /api/v1/audit/report."""

    def test_audit_report_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/audit/report", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "report" in body

    def test_audit_report_invalid_date(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/audit/report",
                {"start_date": "invalid"},
                http,
                method="GET",
            )

        assert _status(result) == 400


class TestVerifyIntegrity:
    """Test GET /api/v1/audit/verify."""

    def test_verify_integrity_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/audit/verify", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is True
        assert body["error_count"] == 0

    def test_verify_integrity_with_errors(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        mock_audit_log.verify_integrity = AsyncMock(
            return_value=(False, ["hash mismatch at entry 3"])
        )
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/audit/verify", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is False
        assert body["error_count"] == 1


class TestActorHistory:
    """Test GET /api/v1/audit/actor/{id}/history."""

    def test_actor_history_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/audit/actor/user-001/history", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["actor_id"] == "user-001"
        assert "entries" in body

    def test_actor_history_custom_days(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/audit/actor/user-001/history",
                {"days": "7"},
                http,
                method="GET",
            )

        assert _status(result) == 200
        assert _body(result)["days"] == 7


class TestResourceHistory:
    """Test GET /api/v1/audit/resource/{id}/history."""

    def test_resource_history_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/audit/resource/res-001/history", {}, http, method="GET"
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["resource_id"] == "res-001"


class TestDeniedAccess:
    """Test GET /api/v1/audit/denied."""

    def test_denied_access_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/audit/denied", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "denied_attempts" in body


# ---------------------------------------------------------------------------
# Routing tests (404 paths, method dispatch)
# ---------------------------------------------------------------------------


class TestRouting:
    """Test route dispatch and 404 handling."""

    def test_unknown_workspace_subpath(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/unknown/deep/path",
                {},
                http,
                method="GET",
            )

        assert _status(result) == 404

    def test_unknown_retention_subpath(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/retention/unknown", {}, http, method="GET")

        assert _status(result) == 404

    def test_unknown_classify_subpath(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/classify/unknown/deep", {}, http, method="GET")

        assert _status(result) == 404

    def test_unknown_audit_subpath(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="PATCH")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/audit/nonexistent", {}, http, method="PATCH")

        assert _status(result) == 404

    def test_handle_returns_none_for_unmatched_prefix(self, handler):
        """handle() returns None when path doesn't match any prefix."""
        _patch_handler(handler, MagicMock(), MagicMock(), MagicMock(), MagicMock())
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/billing", {}, http, method="GET")

        assert result is None

    def test_unknown_invite_subpath(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/invites/unknown", {}, http, method="GET")

        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Invite management tests
# ---------------------------------------------------------------------------


class TestCreateInvite:
    """Test POST /api/v1/workspaces/{id}/invites."""

    @patch("aragora.server.handlers.workspace.invites.emit_handler_event")
    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_create_invite_success(
        self,
        mock_store_fn,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        store = MagicMock()
        store.check_existing.return_value = None
        mock_invite = MagicMock()
        mock_invite.id = "inv-001"
        mock_invite.token = "test-token"
        mock_invite.expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        mock_invite.to_dict.return_value = {
            "id": "inv-001",
            "workspace_id": "ws-001",
            "email": "new@test.com",
            "role": "member",
            "status": "pending",
            "created_by": "user-001",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": mock_invite.expires_at.isoformat(),
            "accepted_by": None,
            "accepted_at": None,
        }
        store.create.return_value = mock_invite
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"email": "new@test.com", "role": "member"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001/invites", {}, http, method="POST")

        assert _status(result) == 201

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_create_invite_missing_email(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001/invites", {}, http, method="POST")

        assert _status(result) == 400
        assert "email" in _body(result).get("error", "").lower()

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_create_invite_invalid_role(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"email": "user@test.com", "role": "superadmin"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001/invites", {}, http, method="POST")

        assert _status(result) == 400
        assert "Invalid role" in _body(result).get("error", "")

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_create_invite_duplicate(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        store = MagicMock()
        store.check_existing.return_value = MagicMock()  # existing invite found
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"email": "dup@test.com", "role": "member"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001/invites", {}, http, method="POST")

        assert _status(result) == 409


class TestCancelInvite:
    """Test DELETE /api/v1/workspaces/{id}/invites/{invite_id}."""

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_cancel_invite_success(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        from aragora.server.handlers.workspace.invites import InviteStatus

        store = MagicMock()
        invite = MagicMock()
        invite.workspace_id = "ws-001"
        invite.status = InviteStatus.PENDING
        invite.email = "user@test.com"
        store.get.return_value = invite
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/invites/inv-001",
                {},
                http,
                method="DELETE",
            )

        assert _status(result) == 200

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_cancel_invite_not_found(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        store = MagicMock()
        store.get.return_value = None
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/invites/inv-missing",
                {},
                http,
                method="DELETE",
            )

        assert _status(result) == 404

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_cancel_invite_wrong_workspace(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        store = MagicMock()
        invite = MagicMock()
        invite.workspace_id = "ws-other"  # Different workspace
        store.get.return_value = invite
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/invites/inv-001",
                {},
                http,
                method="DELETE",
            )

        assert _status(result) == 404

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_cancel_invite_already_accepted(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        from aragora.server.handlers.workspace.invites import InviteStatus

        store = MagicMock()
        invite = MagicMock()
        invite.workspace_id = "ws-001"
        invite.status = InviteStatus.ACCEPTED
        store.get.return_value = invite
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/invites/inv-001",
                {},
                http,
                method="DELETE",
            )

        assert _status(result) == 400


class TestAcceptInvite:
    """Test POST /api/v1/invites/{token}/accept."""

    @patch("aragora.server.handlers.workspace.invites.emit_handler_event")
    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_accept_invite_success(
        self,
        mock_store_fn,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        store = MagicMock()
        invite = MagicMock()
        invite.id = "inv-001"
        invite.workspace_id = "ws-001"
        invite.role = "member"
        invite.created_by = "user-001"
        invite.is_valid.return_value = True
        store.get_by_token.return_value = invite
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/invites/test-token/accept", {}, http, method="POST")

        assert _status(result) == 200
        assert "joined" in _body(result).get("message", "").lower()

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_accept_invite_invalid_token(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        store = MagicMock()
        store.get_by_token.return_value = None
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/invites/bad-token/accept", {}, http, method="POST")

        assert _status(result) == 404

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_accept_invite_expired(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        from aragora.server.handlers.workspace.invites import InviteStatus

        store = MagicMock()
        invite = MagicMock()
        invite.is_valid.return_value = False
        invite.status = InviteStatus.EXPIRED
        invite.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        store.get_by_token.return_value = invite
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/invites/expired-token/accept", {}, http, method="POST")

        assert _status(result) == 410


class TestResendInvite:
    """Test POST /api/v1/workspaces/{id}/invites/{invite_id}/resend."""

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_resend_invite_success(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        from aragora.server.handlers.workspace.invites import InviteStatus

        store = MagicMock()
        invite = MagicMock()
        invite.workspace_id = "ws-001"
        invite.status = InviteStatus.PENDING
        invite.email = "user@test.com"
        invite.expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        store.get.return_value = invite
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/invites/inv-001/resend",
                {},
                http,
                method="POST",
            )

        assert _status(result) == 200
        assert "resent" in _body(result).get("message", "").lower()

    @patch("aragora.server.handlers.workspace.invites.get_invite_store")
    def test_resend_invite_not_found(
        self,
        mock_store_fn,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        store = MagicMock()
        store.get.return_value = None
        mock_store_fn.return_value = store

        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/invites/inv-missing/resend",
                {},
                http,
                method="POST",
            )

        assert _status(result) == 404


# ---------------------------------------------------------------------------
# RBAC profiles tests
# ---------------------------------------------------------------------------


class TestListProfiles:
    """Test GET /api/v1/workspaces/profiles."""

    def test_list_profiles_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        mock_profile = MagicMock()
        mock_profile.value = "lite"
        mock_config = MagicMock()
        mock_config.name = "Lite"
        mock_config.description = "Lite profile"
        mock_config.roles = ["owner", "member"]
        mock_config.default_role = "member"
        mock_config.features = {"basic"}

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", True),
            patch(
                "aragora.server.handlers.workspace_module.RBACProfile",
                [mock_profile],
            ),
            patch(
                "aragora.server.handlers.workspace_module.get_profile_config",
                return_value=mock_config,
            ),
            patch(
                "aragora.server.handlers.workspace_module.get_lite_role_summary",
                return_value={"owner": "Full access", "member": "Basic access"},
            ),
        ):
            result = handler.handle("/api/v1/workspaces/profiles", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "profiles" in body
        assert body["recommended"] == "lite"

    def test_list_profiles_unavailable(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", False),
        ):
            result = handler.handle("/api/v1/workspaces/profiles", {}, http, method="GET")

        assert _status(result) == 503


# ---------------------------------------------------------------------------
# Cache infrastructure tests
# ---------------------------------------------------------------------------


class TestCacheInfrastructure:
    """Test cache invalidation and stats."""

    def test_invalidate_retention_cache_all(self):
        _retention_policy_cache.set("retention:pol-001", {"test": True})
        _retention_policy_cache.set("retention:pol-002", {"test": True})
        count = _invalidate_retention_cache()
        assert count >= 0

    def test_invalidate_retention_cache_by_policy(self):
        _retention_policy_cache.set("retention:pol-001:data", {"test": True})
        count = _invalidate_retention_cache("pol-001")
        assert count >= 0

    def test_invalidate_permission_cache_all(self):
        _permission_cache.set("perm:user:u1:ws:read", (True, ""))
        count = _invalidate_permission_cache()
        assert count >= 0

    def test_invalidate_permission_cache_by_user(self):
        _permission_cache.set("perm:user:u1:ws:read", (True, ""))
        count = _invalidate_permission_cache(user_id="u1")
        assert count >= 0

    def test_invalidate_permission_cache_by_workspace(self):
        _permission_cache.set("perm:ws:ws1:read", (True, ""))
        count = _invalidate_permission_cache(workspace_id="ws1")
        assert count >= 0

    def test_invalidate_audit_cache_all(self):
        _audit_query_cache.set("audit:ws:ws1", {"test": True})
        count = _invalidate_audit_cache()
        assert count >= 0

    def test_invalidate_audit_cache_by_workspace(self):
        _audit_query_cache.set("audit:ws:ws1:data", {"test": True})
        count = _invalidate_audit_cache("ws1")
        assert count >= 0

    def test_cache_stats(self):
        stats = get_workspace_cache_stats()
        assert "retention_policy_cache" in stats
        assert "permission_cache" in stats
        assert "audit_query_cache" in stats


# ---------------------------------------------------------------------------
# handle_post, handle_delete, handle_put delegation tests
# ---------------------------------------------------------------------------


class TestMethodHandlers:
    """Test handle_post, handle_delete, handle_put delegation."""

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_handle_post_delegates_to_handle(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"name": "WS"}, method="POST")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle_post("/api/v1/workspaces", {}, http)

        assert _status(result) == 201

    @patch("aragora.server.handlers.workspace.crud.emit_handler_event")
    def test_handle_delete_delegates_to_handle(
        self,
        mock_emit,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle_delete("/api/v1/workspaces/ws-001", {}, http)

        assert _status(result) == 200

    def test_handle_put_delegates_to_handle(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"name": "Updated"}, method="PUT")

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle_put("/api/v1/retention/policies/pol-001", {}, http)

        assert _status(result) == 200


# ---------------------------------------------------------------------------
# RBAC permission check tests
# ---------------------------------------------------------------------------


class TestRBACPermissionCheck:
    """Test _check_rbac_permission method."""

    def test_rbac_check_returns_none_when_unavailable(self, handler):
        """When RBAC is not available and fail-open, returns None."""
        with (
            patch("aragora.server.handlers.workspace_module.RBAC_AVAILABLE", False),
            patch(
                "aragora.server.handlers.workspace_module.rbac_fail_closed",
                return_value=False,
            ),
        ):
            http = MockHTTPHandler(method="GET")
            # Bypass the patched version to test real _check_rbac_permission
            result = WorkspaceHandler._check_rbac_permission(handler, http, "workspace:read")
            assert result is None

    def test_rbac_check_fail_closed_returns_503(self, handler):
        """When RBAC is not available and fail-closed, returns 503."""
        with (
            patch("aragora.server.handlers.workspace_module.RBAC_AVAILABLE", False),
            patch(
                "aragora.server.handlers.workspace_module.rbac_fail_closed",
                return_value=True,
            ),
        ):
            http = MockHTTPHandler(method="GET")
            result = WorkspaceHandler._check_rbac_permission(handler, http, "workspace:read")
            assert _status(result) == 503


# ---------------------------------------------------------------------------
# Workspace roles tests
# ---------------------------------------------------------------------------


class TestGetWorkspaceRoles:
    """Test GET /api/v1/workspaces/{id}/roles."""

    def test_get_workspace_roles_profiles_unavailable(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", False),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001/roles", {}, http, method="GET")

        assert _status(result) == 503

    def test_get_workspace_roles_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")

        mock_config = MagicMock()
        mock_config.name = "Lite"
        mock_config.roles = ["owner", "member"]

        mock_role = MagicMock()
        mock_role.name = "Owner"
        mock_role.description = "Full access"

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", True),
            patch(
                "aragora.server.handlers.workspace_module.get_profile_config",
                return_value=mock_config,
            ),
            patch(
                "aragora.server.handlers.workspace_module.get_profile_roles",
                return_value={"owner": mock_role, "member": mock_role},
            ),
            patch(
                "aragora.server.handlers.workspace_module.get_available_roles_for_assignment",
                return_value=["member"],
            ),
        ):
            result = handler.handle("/api/v1/workspaces/ws-001/roles", {}, http, method="GET")

        assert _status(result) == 200
        body = _body(result)
        assert "roles" in body
        assert body["workspace_id"] == "ws-001"


# ---------------------------------------------------------------------------
# Update member role tests
# ---------------------------------------------------------------------------


class TestUpdateMemberRole:
    """Test PUT /api/v1/workspaces/{id}/members/{user_id}/role."""

    def test_update_member_role_profiles_unavailable(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"role": "admin"}, method="PUT")

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", False),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/members/user-002/role",
                {},
                http,
                method="PUT",
            )

        assert _status(result) == 503

    def test_update_member_role_missing_role(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={}, method="PUT")

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", True),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/members/user-002/role",
                {},
                http,
                method="PUT",
            )

        assert _status(result) == 400
        assert "role is required" in _body(result).get("error", "")

    def test_update_member_role_success(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        ws = MockWorkspace(member_roles={"user-001": "owner", "user-002": "member"})
        mock_isolation_manager.get_workspace = AsyncMock(return_value=ws)
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"role": "member"}, method="PUT")

        mock_config = MagicMock()
        mock_config.roles = ["owner", "admin", "member"]

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", True),
            patch(
                "aragora.server.handlers.workspace_module.get_profile_config",
                return_value=mock_config,
            ),
            patch(
                "aragora.server.handlers.workspace_module.get_available_roles_for_assignment",
                return_value=["admin", "member"],
            ),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/members/user-002/role",
                {},
                http,
                method="PUT",
            )

        assert _status(result) == 200
        assert "member" in _body(result).get("new_role", "")

    def test_update_member_role_invalid_role_for_profile(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        ws = MockWorkspace(member_roles={"user-001": "owner", "user-002": "member"})
        mock_isolation_manager.get_workspace = AsyncMock(return_value=ws)
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(body={"role": "superadmin"}, method="PUT")

        mock_config = MagicMock()
        mock_config.roles = ["owner", "admin", "member"]

        with (
            patch(
                "aragora.server.handlers.workspace_module.extract_user_from_request",
                return_value=_make_auth_ctx(),
            ),
            patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", True),
            patch(
                "aragora.server.handlers.workspace_module.get_profile_config",
                return_value=mock_config,
            ),
        ):
            result = handler.handle(
                "/api/v1/workspaces/ws-001/members/user-002/role",
                {},
                http,
                method="PUT",
            )

        assert _status(result) == 400
        assert "not available" in _body(result).get("error", "")


# ---------------------------------------------------------------------------
# Handler command override test
# ---------------------------------------------------------------------------


class TestCommandOverride:
    """Test that handler.command overrides method parameter."""

    def test_command_attribute_overrides_method(
        self,
        handler,
        mock_isolation_manager,
        mock_retention_manager,
        mock_audit_log,
        mock_classifier,
    ):
        _patch_handler(
            handler,
            mock_isolation_manager,
            mock_retention_manager,
            mock_audit_log,
            mock_classifier,
        )
        http = MockHTTPHandler(method="GET")
        # Set handler.command to override method
        http.command = "POST"

        with patch(
            "aragora.server.handlers.workspace_module.extract_user_from_request",
            return_value=_make_auth_ctx(),
        ):
            result = handler.handle("/api/v1/workspaces", {}, http, method="GET")

        # Should behave as POST (create workspace), which requires name
        assert _status(result) == 400
        assert "name is required" in _body(result).get("error", "")
