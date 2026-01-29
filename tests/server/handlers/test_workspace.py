"""
Tests for WorkspaceHandler - Workspace and Privacy Management API endpoints.

Tests cover:
- Workspace CRUD operations (create, list, get, delete)
- Member management (add, remove, update role)
- RBAC profiles and workspace roles
- Retention policy management (create, list, get, update, delete, execute)
- Sensitivity classification
- Privacy audit log queries and reports
- Authentication and authorization checks
- Error handling and edge cases
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

            raise AccessDeniedException(f"Workspace {workspace_id} not found")
        return self.workspaces[workspace_id]

    async def delete_workspace(
        self, workspace_id: str, deleted_by: str, force: bool = False
    ) -> None:
        if workspace_id not in self.workspaces:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(f"Workspace {workspace_id} not found")
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

            raise AccessDeniedException(f"Workspace {workspace_id} not found")
        self.workspaces[workspace_id].members.append(user_id)

    async def remove_member(self, workspace_id: str, user_id: str, removed_by: str) -> None:
        if workspace_id not in self.workspaces:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(f"Workspace {workspace_id} not found")
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
            action=kwargs.get("action", "unknown"),
            outcome=kwargs.get("outcome", "success"),
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
# can_handle Tests
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_workspaces_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/workspaces") is True

    def test_handles_workspace_by_id(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/workspaces/workspace-001") is True

    def test_handles_workspace_members(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/workspaces/workspace-001/members") is True

    def test_handles_retention_policies(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/retention/policies") is True

    def test_handles_retention_expiring(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/retention/expiring") is True

    def test_handles_classify(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/classify") is True

    def test_handles_audit_entries(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/audit/entries") is True

    def test_handles_audit_report(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/audit/report") is True

    def test_handles_audit_verify(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/audit/verify") is True

    def test_does_not_handle_other_routes(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/auth/login") is False
        assert workspace_handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# Workspace CRUD Tests
# ===========================================================================


class TestCreateWorkspace:
    """Tests for workspace creation."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_workspace_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST", {"name": "New Workspace"})

        result = workspace_handler.handle("/api/v1/workspaces", {}, mock_handler, "POST")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_workspace_requires_name(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {})

        result = workspace_handler.handle("/api/v1/workspaces", {}, mock_handler, "POST")

        assert get_status(result) == 400
        body = get_body(result)
        assert "name" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_workspace_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"name": "New Workspace"})

        result = workspace_handler.handle("/api/v1/workspaces", {}, mock_handler, "POST")

        assert get_status(result) == 201
        body = get_body(result)
        assert "workspace" in body
        assert body["workspace"]["name"] == "New Workspace"

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_workspace_blocks_cross_tenant(
        self, mock_extract, workspace_handler, auth_context
    ):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "POST", {"name": "New Workspace", "organization_id": "other-org"}
        )

        result = workspace_handler.handle("/api/v1/workspaces", {}, mock_handler, "POST")

        assert get_status(result) == 403
        body = get_body(result)
        assert "another organization" in body.get("error", "").lower()


class TestListWorkspaces:
    """Tests for listing workspaces."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_list_workspaces_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/workspaces", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_list_workspaces_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/workspaces", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "workspaces" in body
        assert "total" in body
        assert body["total"] >= 1

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_list_workspaces_blocks_cross_tenant(
        self, mock_extract, workspace_handler, auth_context
    ):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/workspaces", {"organization_id": "other-org"}, mock_handler, "GET"
        )

        assert get_status(result) == 403


class TestGetWorkspace:
    """Tests for getting a specific workspace."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_get_workspace_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001", {}, mock_handler, "GET"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_get_workspace_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001", {}, mock_handler, "GET"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "workspace" in body
        assert body["workspace"]["id"] == "workspace-001"

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_get_workspace_not_found(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/workspaces/nonexistent", {}, mock_handler, "GET")

        assert get_status(result) == 403  # AccessDeniedException returns 403


class TestDeleteWorkspace:
    """Tests for workspace deletion."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_delete_workspace_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("DELETE")

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001", {}, mock_handler, "DELETE"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_delete_workspace_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("DELETE")

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001", {}, mock_handler, "DELETE"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "deleted" in body.get("message", "").lower()


# ===========================================================================
# Member Management Tests
# ===========================================================================


class TestAddMember:
    """Tests for adding workspace members."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_add_member_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST", {"user_id": "user-456"})

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001/members", {}, mock_handler, "POST"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_add_member_requires_user_id(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {})

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001/members", {}, mock_handler, "POST"
        )

        assert get_status(result) == 400
        body = get_body(result)
        assert "user_id" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_add_member_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "POST", {"user_id": "user-456", "permissions": ["read", "write"]}
        )

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001/members", {}, mock_handler, "POST"
        )

        assert get_status(result) == 201
        body = get_body(result)
        assert "user-456" in body.get("message", "")


class TestRemoveMember:
    """Tests for removing workspace members."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_remove_member_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("DELETE")

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001/members/user-456", {}, mock_handler, "DELETE"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_remove_member_success(self, mock_extract, workspace_handler, auth_context):
        # First add the member
        workspace_handler._isolation_manager.workspaces["workspace-001"].members.append("user-456")

        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("DELETE")

        result = workspace_handler.handle(
            "/api/v1/workspaces/workspace-001/members/user-456", {}, mock_handler, "DELETE"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "removed" in body.get("message", "").lower()


# ===========================================================================
# RBAC Profile Tests
# ===========================================================================


class TestListProfiles:
    """Tests for listing RBAC profiles."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    @patch("aragora.server.handlers.workspace.PROFILES_AVAILABLE", True)
    def test_list_profiles_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/workspaces/profiles", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    @patch("aragora.server.handlers.workspace.PROFILES_AVAILABLE", False)
    def test_list_profiles_unavailable(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/workspaces/profiles", {}, mock_handler, "GET")

        assert get_status(result) == 503


# ===========================================================================
# Retention Policy Tests
# ===========================================================================


class TestListPolicies:
    """Tests for listing retention policies."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_list_policies_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/retention/policies", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_list_policies_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/retention/policies", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "policies" in body
        assert "total" in body


class TestCreatePolicy:
    """Tests for creating retention policies."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_policy_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST", {"name": "New Policy"})

        result = workspace_handler.handle("/api/v1/retention/policies", {}, mock_handler, "POST")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_policy_requires_name(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {})

        result = workspace_handler.handle("/api/v1/retention/policies", {}, mock_handler, "POST")

        assert get_status(result) == 400
        body = get_body(result)
        assert "name" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_policy_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "POST", {"name": "New Policy", "retention_days": 30, "action": "delete"}
        )

        result = workspace_handler.handle("/api/v1/retention/policies", {}, mock_handler, "POST")

        assert get_status(result) == 201
        body = get_body(result)
        assert "policy" in body
        assert body["policy"]["name"] == "New Policy"

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_policy_invalid_action(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"name": "Policy", "action": "invalid_action"})

        result = workspace_handler.handle("/api/v1/retention/policies", {}, mock_handler, "POST")

        assert get_status(result) == 400


class TestGetPolicy:
    """Tests for getting a specific policy."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_get_policy_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001", {}, mock_handler, "GET"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_get_policy_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001", {}, mock_handler, "GET"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "policy" in body
        assert body["policy"]["id"] == "policy-001"

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_get_policy_not_found(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/retention/policies/nonexistent", {}, mock_handler, "GET"
        )

        assert get_status(result) == 404


class TestUpdatePolicy:
    """Tests for updating retention policies."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_update_policy_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("PUT", {"retention_days": 60})

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001", {}, mock_handler, "PUT"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_update_policy_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("PUT", {"retention_days": 60})

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001", {}, mock_handler, "PUT"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "policy" in body


class TestDeletePolicy:
    """Tests for deleting retention policies."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_delete_policy_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("DELETE")

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001", {}, mock_handler, "DELETE"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_delete_policy_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("DELETE")

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001", {}, mock_handler, "DELETE"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "deleted" in body.get("message", "").lower()


class TestExecutePolicy:
    """Tests for executing retention policies."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_execute_policy_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST")

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001/execute", {}, mock_handler, "POST"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_execute_policy_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST")

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001/execute", {}, mock_handler, "POST"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "report" in body
        assert body.get("dry_run") is False

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_execute_policy_dry_run(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST")

        result = workspace_handler.handle(
            "/api/v1/retention/policies/policy-001/execute",
            {"dry_run": "true"},
            mock_handler,
            "POST",
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("dry_run") is True


class TestExpiringItems:
    """Tests for getting expiring items."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_expiring_items_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/retention/expiring", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_expiring_items_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/retention/expiring", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "expiring" in body
        assert "total" in body
        assert "days_ahead" in body


# ===========================================================================
# Classification Tests
# ===========================================================================


class TestClassifyContent:
    """Tests for content classification."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_classify_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST", {"content": "Test content"})

        result = workspace_handler.handle("/api/v1/classify", {}, mock_handler, "POST")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_classify_requires_content(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {})

        result = workspace_handler.handle("/api/v1/classify", {}, mock_handler, "POST")

        assert get_status(result) == 400
        body = get_body(result)
        assert "content" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_classify_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"content": "Sensitive data"})

        result = workspace_handler.handle("/api/v1/classify", {}, mock_handler, "POST")

        assert get_status(result) == 200
        body = get_body(result)
        assert "classification" in body
        assert "level" in body["classification"]


class TestGetLevelPolicy:
    """Tests for getting sensitivity level policy."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_get_level_policy_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/classify/policy/internal", {}, mock_handler, "GET"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    @patch("aragora.server.handlers.workspace.SensitivityLevel")
    def test_get_level_policy_success(
        self, mock_level, mock_extract, workspace_handler, auth_context
    ):
        mock_extract.return_value = auth_context
        mock_level.return_value = MagicMock(value="internal")
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/classify/policy/internal", {}, mock_handler, "GET"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "policy" in body


# ===========================================================================
# Audit Log Tests
# ===========================================================================


class TestQueryAudit:
    """Tests for querying audit entries."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_query_audit_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/audit/entries", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_query_audit_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/audit/entries", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "entries" in body
        assert "total" in body


class TestAuditReport:
    """Tests for generating audit reports."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_audit_report_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/audit/report", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_audit_report_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/audit/report", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "report" in body


class TestVerifyIntegrity:
    """Tests for verifying audit log integrity."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_verify_integrity_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/audit/verify", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_verify_integrity_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/audit/verify", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "valid" in body
        assert body["valid"] is True


class TestActorHistory:
    """Tests for getting actor history."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_actor_history_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/audit/actor/user-123/history", {}, mock_handler, "GET"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_actor_history_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/audit/actor/user-123/history", {}, mock_handler, "GET"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "actor_id" in body
        assert "entries" in body


class TestResourceHistory:
    """Tests for getting resource history."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_resource_history_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/audit/resource/workspace-001/history", {}, mock_handler, "GET"
        )

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_resource_history_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/audit/resource/workspace-001/history", {}, mock_handler, "GET"
        )

        assert get_status(result) == 200
        body = get_body(result)
        assert "resource_id" in body
        assert "entries" in body


class TestDeniedAccess:
    """Tests for getting denied access attempts."""

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_denied_access_requires_auth(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/audit/denied", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_denied_access_success(self, mock_extract, workspace_handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle("/api/v1/audit/denied", {}, mock_handler, "GET")

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

        result = workspace_handler.handle("/api/v1/workspaces", {}, handler, "POST")

        assert get_status(result) == 400

    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_workspace_without_org(self, mock_extract, workspace_handler):
        mock_extract.return_value = MockAuthContext(org_id=None)
        mock_handler = create_mock_handler("POST", {"name": "New Workspace"})

        result = workspace_handler.handle("/api/v1/workspaces", {}, mock_handler, "POST")

        assert get_status(result) == 400
        body = get_body(result)
        assert "organization" in body.get("error", "").lower()

    def test_not_found_route(self, workspace_handler):
        mock_handler = create_mock_handler("GET")

        result = workspace_handler.handle(
            "/api/v1/workspaces/ws-001/unknown", {}, mock_handler, "GET"
        )

        assert get_status(result) == 404

    def test_handle_post_delegates(self, workspace_handler):
        """Test that handle_post delegates to handle with POST method."""
        mock_handler = create_mock_handler("POST", {"name": "Test"})

        with patch.object(workspace_handler, "handle") as mock_handle:
            mock_handle.return_value = MagicMock(status_code=200)
            workspace_handler.handle_post("/api/v1/workspaces", {}, mock_handler)
            mock_handle.assert_called_once_with(
                "/api/v1/workspaces", {}, mock_handler, method="POST"
            )

    def test_handle_delete_delegates(self, workspace_handler):
        """Test that handle_delete delegates to handle with DELETE method."""
        mock_handler = create_mock_handler("DELETE")

        with patch.object(workspace_handler, "handle") as mock_handle:
            mock_handle.return_value = MagicMock(status_code=200)
            workspace_handler.handle_delete("/api/v1/workspaces/ws-001", {}, mock_handler)
            mock_handle.assert_called_once_with(
                "/api/v1/workspaces/ws-001", {}, mock_handler, method="DELETE"
            )

    def test_handle_put_delegates(self, workspace_handler):
        """Test that handle_put delegates to handle with PUT method."""
        mock_handler = create_mock_handler("PUT", {"retention_days": 60})

        with patch.object(workspace_handler, "handle") as mock_handle:
            mock_handle.return_value = MagicMock(status_code=200)
            workspace_handler.handle_put("/api/v1/retention/policies/p-001", {}, mock_handler)
            mock_handle.assert_called_once_with(
                "/api/v1/retention/policies/p-001", {}, mock_handler, method="PUT"
            )


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

    def test_audit_log_lazy_init(self, workspace_handler):
        """Test that audit log is lazily initialized."""
        workspace_handler._audit_log = None

        audit_log = workspace_handler._get_audit_log()

        assert audit_log is not None
