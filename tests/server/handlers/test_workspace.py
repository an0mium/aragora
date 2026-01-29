"""
Tests for WorkspaceHandler - Workspace and Privacy Management API endpoints.

Tests cover:
- Workspace CRUD operations (create, list, get, delete)
- Member management (add, remove, update role)
- Retention policy management (create, list, get, update, delete, execute)
- Sensitivity classification
- Privacy audit log queries and reports
- Authentication and authorization checks
- Error handling and edge cases

Note: Routing tests are in test_workspace_handler.py. This file focuses on
endpoint behavior with mocked dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.workspace import WorkspaceHandler


# ===========================================================================
# Rate Limit Bypass for Testing
# ===========================================================================


def _always_allowed(key: str) -> bool:
    """Always allow requests for testing."""
    return True


@pytest.fixture(autouse=True)
def disable_rate_limits():
    """Disable rate limits for all tests in this module."""
    import sys

    rl_module = sys.modules.get("aragora.server.handlers.utils.rate_limit")
    if not rl_module:
        yield
        return

    # Patch all existing limiters to always allow
    original_is_allowed = {}
    if hasattr(rl_module, "_limiters"):
        for name, limiter in rl_module._limiters.items():
            original_is_allowed[name] = limiter.is_allowed
            limiter.is_allowed = _always_allowed

    yield

    # Restore original is_allowed methods
    if hasattr(rl_module, "_limiters"):
        for name, original in original_is_allowed.items():
            if name in rl_module._limiters:
                rl_module._limiters[name].is_allowed = original


@pytest.fixture(autouse=True)
def bypass_rbac():
    """Bypass RBAC permission checks for all tests in this module."""
    with patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", False):
        yield


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: str | None = "org-123"
    role: str = "admin"
    is_active: bool = True


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "admin"


@dataclass
class MockWorkspace:
    """Mock workspace for testing."""

    id: str = "workspace-001"
    name: str = "Test Workspace"
    organization_id: str = "org-123"
    created_by: str = "user-123"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    members: List[str] = field(default_factory=lambda: ["user-123"])
    rbac_profile: str = "lite"
    member_roles: Dict[str, str] = field(default_factory=lambda: {"user-123": "owner"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "organization_id": self.organization_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "members": self.members,
            "rbac_profile": self.rbac_profile,
            "member_roles": self.member_roles,
        }


@dataclass
class MockRetentionPolicy:
    """Mock retention policy for testing."""

    id: str = "policy-001"
    name: str = "Default Policy"
    description: str = "Test retention policy"
    retention_days: int = 90
    action: Any = field(default_factory=lambda: MagicMock(value="delete"))
    enabled: bool = True
    applies_to: List[str] = field(default_factory=lambda: ["documents", "findings"])
    workspace_ids: Optional[List[str]] = None
    grace_period_days: int = 7
    notify_before_days: int = 7
    exclude_sensitivity_levels: List[str] = field(default_factory=list)
    exclude_tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: Optional[datetime] = None


@dataclass
class MockRetentionReport:
    """Mock retention execution report."""

    items_evaluated: int = 100
    items_deleted: int = 25
    items_archived: int = 0
    items_anonymized: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items_evaluated": self.items_evaluated,
            "items_deleted": self.items_deleted,
            "items_archived": self.items_archived,
            "items_anonymized": self.items_anonymized,
            "errors": self.errors,
        }


@dataclass
class MockClassificationResult:
    """Mock classification result."""

    level: Any = field(default_factory=lambda: MagicMock(value="internal"))
    confidence: float = 0.85
    indicators: List[str] = field(default_factory=lambda: ["keyword_match"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "confidence": self.confidence,
            "indicators": self.indicators,
        }


@dataclass
class MockAuditEntry:
    """Mock audit entry."""

    id: str = "audit-001"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action: str = "create_workspace"
    outcome: str = "success"
    actor_id: str = "user-123"
    resource_id: str = "workspace-001"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "outcome": self.outcome,
            "actor_id": self.actor_id,
            "resource_id": self.resource_id,
        }


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: Dict[str, MockUser] = {"user-123": MockUser()}

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)


class MockIsolationManager:
    """Mock data isolation manager."""

    def __init__(self):
        self.workspaces: Dict[str, MockWorkspace] = {
            "workspace-001": MockWorkspace(),
        }

    async def create_workspace(
        self,
        organization_id: str,
        name: str,
        created_by: str,
        initial_members: List[str] = None,
    ) -> MockWorkspace:
        workspace = MockWorkspace(
            id=f"workspace-{len(self.workspaces) + 1:03d}",
            name=name,
            organization_id=organization_id,
            created_by=created_by,
            members=initial_members or [created_by],
        )
        self.workspaces[workspace.id] = workspace
        return workspace

    async def list_workspaces(self, actor: str, organization_id: str = None) -> List[MockWorkspace]:
        return list(self.workspaces.values())

    async def get_workspace(self, workspace_id: str, actor: str) -> MockWorkspace:
        if workspace_id not in self.workspaces:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                message=f"Workspace {workspace_id} not found",
                workspace_id=workspace_id,
                actor=actor,
                action="read",
            )
        return self.workspaces[workspace_id]

    async def delete_workspace(
        self, workspace_id: str, deleted_by: str, force: bool = False
    ) -> None:
        if workspace_id not in self.workspaces:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                message=f"Workspace {workspace_id} not found",
                workspace_id=workspace_id,
                actor=deleted_by,
                action="delete",
            )
        del self.workspaces[workspace_id]

    async def add_member(
        self,
        workspace_id: str,
        user_id: str,
        permissions: List[Any],
        added_by: str,
    ) -> None:
        if workspace_id not in self.workspaces:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                message=f"Workspace {workspace_id} not found",
                workspace_id=workspace_id,
                actor=added_by,
                action="add_member",
            )
        self.workspaces[workspace_id].members.append(user_id)

    async def remove_member(self, workspace_id: str, user_id: str, removed_by: str) -> None:
        if workspace_id not in self.workspaces:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                message=f"Workspace {workspace_id} not found",
                workspace_id=workspace_id,
                actor=removed_by,
                action="remove_member",
            )
        if user_id in self.workspaces[workspace_id].members:
            self.workspaces[workspace_id].members.remove(user_id)


class MockRetentionManager:
    """Mock retention policy manager."""

    def __init__(self):
        self.policies: Dict[str, MockRetentionPolicy] = {
            "policy-001": MockRetentionPolicy(),
        }

    def list_policies(self, workspace_id: str = None) -> List[MockRetentionPolicy]:
        return list(self.policies.values())

    def get_policy(self, policy_id: str) -> MockRetentionPolicy | None:
        return self.policies.get(policy_id)

    def create_policy(
        self,
        name: str,
        retention_days: int,
        action: Any,
        workspace_ids: List[str] = None,
        description: str = "",
        applies_to: List[str] = None,
    ) -> MockRetentionPolicy:
        policy = MockRetentionPolicy(
            id=f"policy-{len(self.policies) + 1:03d}",
            name=name,
            description=description,
            retention_days=retention_days,
            action=action,
            applies_to=applies_to or ["documents"],
            workspace_ids=workspace_ids,
        )
        self.policies[policy.id] = policy
        return policy

    def update_policy(self, policy_id: str, **kwargs) -> MockRetentionPolicy:
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")
        policy = self.policies[policy_id]
        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        return policy

    def delete_policy(self, policy_id: str) -> None:
        if policy_id in self.policies:
            del self.policies[policy_id]

    async def execute_policy(self, policy_id: str, dry_run: bool = False) -> MockRetentionReport:
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")
        return MockRetentionReport()

    async def check_expiring_soon(
        self, workspace_id: str = None, days: int = 14
    ) -> List[Dict[str, Any]]:
        return [
            {
                "id": "doc-001",
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(),
            },
            {
                "id": "doc-002",
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat(),
            },
        ]


class MockClassifier:
    """Mock sensitivity classifier."""

    async def classify(
        self, content: str, document_id: str = "", metadata: Dict = None
    ) -> MockClassificationResult:
        return MockClassificationResult()

    def get_level_policy(self, level: Any) -> Dict[str, Any]:
        return {
            "encryption_required": True,
            "retention_max_days": 365,
            "export_allowed": False,
        }


class MockAuditLog:
    """Mock privacy audit log."""

    def __init__(self):
        self.entries: List[MockAuditEntry] = [MockAuditEntry()]

    async def log(self, **kwargs) -> None:
        entry = MockAuditEntry(
            id=f"audit-{len(self.entries) + 1:03d}",
            action=str(kwargs.get("action", "unknown")),
            outcome=str(kwargs.get("outcome", "success")),
        )
        self.entries.append(entry)

    async def query(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        actor_id: str = None,
        resource_id: str = None,
        workspace_id: str = None,
        action: Any = None,
        outcome: Any = None,
        limit: int = 100,
    ) -> List[MockAuditEntry]:
        return self.entries[:limit]

    async def generate_compliance_report(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        workspace_id: str = None,
        format: str = "json",
    ) -> Dict[str, Any]:
        return {
            "report_id": "report-001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(self.entries),
            "summary": {"success": 10, "denied": 2, "failed": 0},
        }

    async def verify_integrity(
        self, start_date: datetime = None, end_date: datetime = None
    ) -> tuple[bool, List[str]]:
        return True, []

    async def get_actor_history(self, actor_id: str, days: int = 30) -> List[MockAuditEntry]:
        return [e for e in self.entries if e.actor_id == actor_id]

    async def get_resource_history(self, resource_id: str, days: int = 30) -> List[MockAuditEntry]:
        return [e for e in self.entries if e.resource_id == resource_id]

    async def get_denied_access_attempts(self, days: int = 7) -> List[MockAuditEntry]:
        return [e for e in self.entries if e.outcome == "denied"]


# ===========================================================================
# Helper Functions
# ===========================================================================


def get_status(result) -> int:
    """Extract status code from HandlerResult or tuple."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body from HandlerResult or tuple."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    body = result[0]
    if isinstance(body, dict):
        return body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    return json.loads(body)


def create_mock_handler(method: str = "GET", body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = {"Content-Type": "application/json"}

    if body:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.rfile = BytesIO(body_bytes)
        handler.headers["Content-Length"] = str(len(body_bytes))
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def user_store():
    """Create a mock user store with test data."""
    return MockUserStore()


@pytest.fixture
def isolation_manager():
    """Create a mock isolation manager."""
    return MockIsolationManager()


@pytest.fixture
def retention_manager():
    """Create a mock retention manager."""
    return MockRetentionManager()


@pytest.fixture
def classifier():
    """Create a mock classifier."""
    return MockClassifier()


@pytest.fixture
def audit_log():
    """Create a mock audit log."""
    return MockAuditLog()


@pytest.fixture
def workspace_handler(user_store, isolation_manager, retention_manager, classifier, audit_log):
    """Create a workspace handler with mocked dependencies."""
    ctx = {"user_store": user_store}
    handler = WorkspaceHandler(ctx)
    handler._isolation_manager = isolation_manager
    handler._retention_manager = retention_manager
    handler._classifier = classifier
    handler._audit_log = audit_log
    return handler


@pytest.fixture
def auth_context():
    """Create a mock auth context."""
    return MockAuthContext()


# ===========================================================================
# Workspace CRUD Handler Method Tests
# ===========================================================================


class TestHandleCreateWorkspace:
    """Tests for _handle_create_workspace method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("POST", {"name": "New Workspace"})

        result = workspace_handler._handle_create_workspace(mock_http)

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_name(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST", {})

        result = workspace_handler._handle_create_workspace(mock_http)

        assert get_status(result) == 400
        body = get_body(result)
        assert "name" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_organization(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(org_id=None)
        mock_http = create_mock_handler("POST", {"name": "New Workspace"})

        result = workspace_handler._handle_create_workspace(mock_http)

        assert get_status(result) == 400
        body = get_body(result)
        assert "organization" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST", {"name": "New Workspace"})

        result = workspace_handler._handle_create_workspace(mock_http)

        assert get_status(result) == 201
        body = get_body(result)
        assert "workspace" in body
        assert body["workspace"]["name"] == "New Workspace"

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_blocks_cross_tenant(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler(
            "POST", {"name": "New Workspace", "organization_id": "other-org"}
        )

        result = workspace_handler._handle_create_workspace(mock_http)

        assert get_status(result) == 403
        body = get_body(result)
        assert "another organization" in body.get("error", "").lower()


class TestHandleListWorkspaces:
    """Tests for _handle_list_workspaces method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_list_workspaces(mock_http, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_list_workspaces(mock_http, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "workspaces" in body
        assert "total" in body
        assert body["total"] >= 1

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_blocks_cross_tenant(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_list_workspaces(
            mock_http, {"organization_id": "other-org"}
        )

        assert get_status(result) == 403


class TestHandleGetWorkspace:
    """Tests for _handle_get_workspace method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_get_workspace(mock_http, "workspace-001")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_get_workspace(mock_http, "workspace-001")

        assert get_status(result) == 200
        body = get_body(result)
        assert "workspace" in body
        assert body["workspace"]["id"] == "workspace-001"

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_not_found(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_get_workspace(mock_http, "nonexistent")

        assert get_status(result) == 403  # AccessDeniedException returns 403


class TestHandleDeleteWorkspace:
    """Tests for _handle_delete_workspace method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("DELETE")

        result = workspace_handler._handle_delete_workspace(mock_http, "workspace-001")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("DELETE")

        result = workspace_handler._handle_delete_workspace(mock_http, "workspace-001")

        assert get_status(result) == 200
        body = get_body(result)
        assert "deleted" in body.get("message", "").lower()


# ===========================================================================
# Member Management Handler Method Tests
# ===========================================================================


class TestHandleAddMember:
    """Tests for _handle_add_member method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("POST", {"user_id": "user-456"})

        result = workspace_handler._handle_add_member(mock_http, "workspace-001")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_user_id(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST", {})

        result = workspace_handler._handle_add_member(mock_http, "workspace-001")

        assert get_status(result) == 400
        body = get_body(result)
        assert "user_id" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler(
            "POST", {"user_id": "user-456", "permissions": ["read", "write"]}
        )

        result = workspace_handler._handle_add_member(mock_http, "workspace-001")

        assert get_status(result) == 201
        body = get_body(result)
        assert "user-456" in body.get("message", "")


class TestHandleRemoveMember:
    """Tests for _handle_remove_member method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("DELETE")

        result = workspace_handler._handle_remove_member(mock_http, "workspace-001", "user-456")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        # First add the member
        workspace_handler._isolation_manager.workspaces["workspace-001"].members.append("user-456")

        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("DELETE")

        result = workspace_handler._handle_remove_member(mock_http, "workspace-001", "user-456")

        assert get_status(result) == 200
        body = get_body(result)
        assert "removed" in body.get("message", "").lower()


# ===========================================================================
# Retention Policy Handler Method Tests
# ===========================================================================


class TestHandleListPolicies:
    """Tests for _handle_list_policies method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_list_policies(mock_http, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_list_policies(mock_http, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "policies" in body
        assert "total" in body


class TestHandleCreatePolicy:
    """Tests for _handle_create_policy method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("POST", {"name": "New Policy"})

        result = workspace_handler._handle_create_policy(mock_http)

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_name(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST", {})

        result = workspace_handler._handle_create_policy(mock_http)

        assert get_status(result) == 400
        body = get_body(result)
        assert "name" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler(
            "POST", {"name": "New Policy", "retention_days": 30, "action": "delete"}
        )

        result = workspace_handler._handle_create_policy(mock_http)

        assert get_status(result) == 201
        body = get_body(result)
        assert "policy" in body
        assert body["policy"]["name"] == "New Policy"

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_invalid_action(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST", {"name": "Policy", "action": "invalid_action"})

        result = workspace_handler._handle_create_policy(mock_http)

        assert get_status(result) == 400


class TestHandleGetPolicy:
    """Tests for _handle_get_policy method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_get_policy(mock_http, "policy-001")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_get_policy(mock_http, "policy-001")

        assert get_status(result) == 200
        body = get_body(result)
        assert "policy" in body
        assert body["policy"]["id"] == "policy-001"

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_not_found(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_get_policy(mock_http, "nonexistent")

        assert get_status(result) == 404


class TestHandleUpdatePolicy:
    """Tests for _handle_update_policy method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("PUT", {"retention_days": 60})

        result = workspace_handler._handle_update_policy(mock_http, "policy-001")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("PUT", {"retention_days": 60})

        result = workspace_handler._handle_update_policy(mock_http, "policy-001")

        assert get_status(result) == 200
        body = get_body(result)
        assert "policy" in body


class TestHandleDeletePolicy:
    """Tests for _handle_delete_policy method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("DELETE")

        result = workspace_handler._handle_delete_policy(mock_http, "policy-001")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("DELETE")

        result = workspace_handler._handle_delete_policy(mock_http, "policy-001")

        assert get_status(result) == 200
        body = get_body(result)
        assert "deleted" in body.get("message", "").lower()


class TestHandleExecutePolicy:
    """Tests for _handle_execute_policy method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("POST")

        result = workspace_handler._handle_execute_policy(mock_http, "policy-001", {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST")

        result = workspace_handler._handle_execute_policy(mock_http, "policy-001", {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "report" in body
        assert body.get("dry_run") is False

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_dry_run(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST")

        result = workspace_handler._handle_execute_policy(
            mock_http, "policy-001", {"dry_run": "true"}
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("dry_run") is True


class TestHandleExpiringItems:
    """Tests for _handle_expiring_items method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_expiring_items(mock_http, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_expiring_items(mock_http, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "expiring" in body
        assert "total" in body
        assert "days_ahead" in body


# ===========================================================================
# Classification Handler Method Tests
# ===========================================================================


class TestHandleClassifyContent:
    """Tests for _handle_classify_content method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("POST", {"content": "Test content"})

        result = workspace_handler._handle_classify_content(mock_http)

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_content(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST", {})

        result = workspace_handler._handle_classify_content(mock_http)

        assert get_status(result) == 400
        body = get_body(result)
        assert "content" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("POST", {"content": "Sensitive data"})

        result = workspace_handler._handle_classify_content(mock_http)

        assert get_status(result) == 200
        body = get_body(result)
        assert "classification" in body
        assert "level" in body["classification"]


class TestHandleGetLevelPolicy:
    """Tests for _handle_get_level_policy method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_get_level_policy(mock_http, "internal")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    @patch("aragora.server.handlers.workspace.SensitivityLevel")
    def test_success(self, mock_level, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_level.return_value = MagicMock(value="internal")
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_get_level_policy(mock_http, "internal")

        assert get_status(result) == 200
        body = get_body(result)
        assert "policy" in body


# ===========================================================================
# Audit Log Handler Method Tests
# ===========================================================================


class TestHandleQueryAudit:
    """Tests for _handle_query_audit method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_query_audit(mock_http, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_query_audit(mock_http, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "entries" in body
        assert "total" in body


class TestHandleAuditReport:
    """Tests for _handle_audit_report method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_audit_report(mock_http, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_audit_report(mock_http, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "report" in body


class TestHandleVerifyIntegrity:
    """Tests for _handle_verify_integrity method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_verify_integrity(mock_http, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_verify_integrity(mock_http, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "valid" in body
        assert body["valid"] is True


class TestHandleActorHistory:
    """Tests for _handle_actor_history method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_actor_history(mock_http, "user-123", {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_actor_history(mock_http, "user-123", {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "actor_id" in body
        assert "entries" in body


class TestHandleResourceHistory:
    """Tests for _handle_resource_history method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_resource_history(mock_http, "workspace-001", {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_resource_history(mock_http, "workspace-001", {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "resource_id" in body
        assert "entries" in body


class TestHandleDeniedAccess:
    """Tests for _handle_denied_access method."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_requires_authentication(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_denied_access(mock_http, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_http = create_mock_handler("GET")

        result = workspace_handler._handle_denied_access(mock_http, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "denied_attempts" in body
        assert "total" in body


# ===========================================================================
# Edge Cases and Error Handling
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_invalid_json_body(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Type": "application/json", "Content-Length": "10"}
        handler.rfile = BytesIO(b"invalid json")

        result = workspace_handler._handle_create_workspace(handler)

        assert get_status(result) == 400

    def test_handler_delegates_correctly(self, workspace_handler):
        """Test that handle_post/handle_delete/handle_put delegate to handle."""
        mock_http = create_mock_handler("POST", {"name": "Test"})

        with patch.object(workspace_handler, "handle") as mock_handle:
            mock_handle.return_value = MagicMock(status_code=200)
            workspace_handler.handle_post("/api/v1/workspaces", {}, mock_http)
            mock_handle.assert_called_once_with("/api/v1/workspaces", {}, mock_http, method="POST")


class TestManagerInitialization:
    """Tests for lazy manager initialization."""

    def test_isolation_manager_lazy_init(self, workspace_handler):
        """Test that isolation manager is lazily initialized."""
        workspace_handler._isolation_manager = None

        manager = workspace_handler._get_isolation_manager()

        assert manager is not None

    def test_retention_manager_lazy_init(self, workspace_handler):
        """Test that retention manager is lazily initialized."""
        workspace_handler._retention_manager = None

        manager = workspace_handler._get_retention_manager()

        assert manager is not None

    def test_classifier_lazy_init(self, workspace_handler):
        """Test that classifier is lazily initialized."""
        workspace_handler._classifier = None

        classifier = workspace_handler._get_classifier()

        assert classifier is not None

    @patch("aragora.server.handlers.workspace.PrivacyAuditLog")
    def test_audit_log_lazy_init(self, mock_audit_log_class, workspace_handler):
        """Test that audit log is lazily initialized."""
        workspace_handler._audit_log = None
        mock_audit_log_class.return_value = MagicMock()

        audit_log = workspace_handler._get_audit_log()

        assert audit_log is not None
