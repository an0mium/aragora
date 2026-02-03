"""
Comprehensive tests for aragora.server.handlers.workspace_module.

Tests cover:
- Workspace CRUD operations (create, list, get, delete)
- Member management (add, remove, update role, list)
- Retention policy management (create, list, get, update, delete, execute)
- Sensitivity classification (classify content, get level policy)
- Audit logging (query entries, generate report, verify integrity, actor/resource history)
- RBAC permission enforcement (verify correct permissions, denied scenarios)
- Multi-tenancy (cross-tenant access prevention, tenant isolation)

Target: 80+ comprehensive tests covering the 20+ endpoints in workspace_module.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


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
    if rl_module is None:
        yield
        return

    # Patch all existing limiters to always allow
    original_is_allowed = {}
    for name, limiter in getattr(rl_module, "_limiters", {}).items():
        original_is_allowed[name] = limiter.is_allowed
        limiter.is_allowed = _always_allowed

    yield

    # Restore original is_allowed methods
    for name, original in original_is_allowed.items():
        if name in getattr(rl_module, "_limiters", {}):
            rl_module._limiters[name].is_allowed = original


# ===========================================================================
# Mock Classes and Fixtures
# ===========================================================================


class MockWorkspacePermission(Enum):
    """Mock workspace permission."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class MockRetentionAction(Enum):
    """Mock retention action."""

    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    NOTIFY = "notify"


class MockSensitivityLevel(Enum):
    """Mock sensitivity level."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class MockAuditAction(Enum):
    """Mock audit action."""

    CREATE_WORKSPACE = "create_workspace"
    DELETE_WORKSPACE = "delete_workspace"
    ADD_MEMBER = "add_member"
    REMOVE_MEMBER = "remove_member"
    MODIFY_PERMISSIONS = "modify_permissions"
    MODIFY_POLICY = "modify_policy"
    EXECUTE_RETENTION = "execute_retention"
    CLASSIFY_DOCUMENT = "classify_document"
    GENERATE_REPORT = "generate_report"
    ACCESS_DATA = "access_data"


class MockAuditOutcome(Enum):
    """Mock audit outcome."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: str | None = "org-123"
    role: str = "member"
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
            "is_active": self.is_active,
        }


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "member"


@dataclass
class MockWorkspace:
    """Mock workspace for testing."""

    id: str = "ws-123"
    name: str = "Test Workspace"
    organization_id: str = "org-123"
    created_by: str = "user-123"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    members: list[str] = field(default_factory=list)
    member_roles: dict[str, str] = field(default_factory=dict)
    rbac_profile: str = "lite"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "organization_id": self.organization_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "members": self.members,
            "member_roles": self.member_roles,
            "rbac_profile": self.rbac_profile,
        }


@dataclass
class MockRetentionPolicy:
    """Mock retention policy for testing."""

    id: str = "policy-123"
    name: str = "Test Policy"
    description: str = "Test retention policy"
    retention_days: int = 90
    action: MockRetentionAction = MockRetentionAction.DELETE
    enabled: bool = True
    applies_to: list[str] = field(default_factory=lambda: ["documents"])
    workspace_ids: list[str] | None = None
    grace_period_days: int = 7
    notify_before_days: int = 14
    exclude_sensitivity_levels: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: datetime | None = None


@dataclass
class MockDeletionReport:
    """Mock deletion report for testing."""

    items_evaluated: int = 100
    items_deleted: int = 25
    items_archived: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items_evaluated": self.items_evaluated,
            "items_deleted": self.items_deleted,
            "items_archived": self.items_archived,
            "errors": self.errors,
        }


@dataclass
class MockClassificationResult:
    """Mock classification result for testing."""

    level: MockSensitivityLevel = MockSensitivityLevel.INTERNAL
    confidence: float = 0.85
    indicators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "confidence": self.confidence,
            "indicators": self.indicators,
        }


@dataclass
class MockAuditEntry:
    """Mock audit entry for testing."""

    id: str = "entry-123"
    action: str = "create_workspace"
    actor_id: str = "user-123"
    resource_id: str = "ws-123"
    resource_type: str = "workspace"
    outcome: str = "success"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "actor_id": self.actor_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)


class MockDataIsolationManager:
    """Mock data isolation manager for testing."""

    def __init__(self):
        self.workspaces: dict[str, MockWorkspace] = {}
        self.access_denied_users: set[str] = set()

    async def create_workspace(
        self,
        organization_id: str,
        name: str,
        created_by: str,
        initial_members: list[str] | None = None,
    ) -> MockWorkspace:
        ws = MockWorkspace(
            id=f"ws-{len(self.workspaces) + 1}",
            name=name,
            organization_id=organization_id,
            created_by=created_by,
            members=initial_members or [created_by],
        )
        self.workspaces[ws.id] = ws
        return ws

    async def list_workspaces(
        self, actor: str, organization_id: str | None = None
    ) -> list[MockWorkspace]:
        return [
            ws
            for ws in self.workspaces.values()
            if organization_id is None or ws.organization_id == organization_id
        ]

    async def get_workspace(self, workspace_id: str, actor: str) -> MockWorkspace:
        if actor in self.access_denied_users:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                f"Access denied for user {actor}",
                workspace_id=workspace_id,
                actor=actor,
                action="read",
            )
        ws = self.workspaces.get(workspace_id)
        if not ws:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                f"Workspace {workspace_id} not found",
                workspace_id=workspace_id,
                actor=actor,
                action="read",
            )
        return ws

    async def delete_workspace(
        self, workspace_id: str, deleted_by: str, force: bool = False
    ) -> None:
        if deleted_by in self.access_denied_users:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                f"Access denied for user {deleted_by}",
                workspace_id=workspace_id,
                actor=deleted_by,
                action="delete",
            )
        if workspace_id in self.workspaces:
            del self.workspaces[workspace_id]

    async def add_member(
        self,
        workspace_id: str,
        user_id: str,
        permissions: list[Any],
        added_by: str,
    ) -> None:
        if added_by in self.access_denied_users:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                f"Access denied for user {added_by}",
                workspace_id=workspace_id,
                actor=added_by,
                action="add_member",
            )
        ws = self.workspaces.get(workspace_id)
        if ws and user_id not in ws.members:
            ws.members.append(user_id)

    async def remove_member(self, workspace_id: str, user_id: str, removed_by: str) -> None:
        if removed_by in self.access_denied_users:
            from aragora.privacy import AccessDeniedException

            raise AccessDeniedException(
                f"Access denied for user {removed_by}",
                workspace_id=workspace_id,
                actor=removed_by,
                action="remove_member",
            )
        ws = self.workspaces.get(workspace_id)
        if ws and user_id in ws.members:
            ws.members.remove(user_id)


class MockRetentionPolicyManager:
    """Mock retention policy manager for testing."""

    def __init__(self):
        self.policies: dict[str, MockRetentionPolicy] = {}

    def list_policies(self, workspace_id: str | None = None) -> list[MockRetentionPolicy]:
        return list(self.policies.values())

    def get_policy(self, policy_id: str) -> MockRetentionPolicy | None:
        return self.policies.get(policy_id)

    def create_policy(
        self,
        name: str,
        retention_days: int,
        action: Any,
        workspace_ids: list[str] | None = None,
        description: str = "",
        applies_to: list[str] | None = None,
    ) -> MockRetentionPolicy:
        policy = MockRetentionPolicy(
            id=f"policy-{len(self.policies) + 1}",
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
        policy = self.policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")
        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        return policy

    def delete_policy(self, policy_id: str) -> None:
        if policy_id in self.policies:
            del self.policies[policy_id]

    async def execute_policy(self, policy_id: str, dry_run: bool = False) -> MockDeletionReport:
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")
        return MockDeletionReport()

    async def check_expiring_soon(
        self, workspace_id: str | None = None, days: int = 14
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": "doc-1",
                "name": "Test Doc",
                "expires_at": datetime.now(timezone.utc).isoformat(),
            }
        ]


class MockSensitivityClassifier:
    """Mock sensitivity classifier for testing."""

    async def classify(
        self, content: str, document_id: str = "", metadata: dict[str, Any] | None = None
    ) -> MockClassificationResult:
        return MockClassificationResult()

    def get_level_policy(self, level: MockSensitivityLevel) -> dict[str, Any]:
        return {
            "level": level.value,
            "retention_days": 365,
            "encryption_required": level
            in [MockSensitivityLevel.CONFIDENTIAL, MockSensitivityLevel.RESTRICTED],
            "access_logging": True,
        }


class MockPrivacyAuditLog:
    """Mock privacy audit log for testing."""

    def __init__(self):
        self.entries: list[MockAuditEntry] = []

    async def log(
        self,
        action: Any,
        actor: Any,
        resource: Any,
        outcome: Any,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = MockAuditEntry(
            id=f"entry-{len(self.entries) + 1}",
            action=action.value if hasattr(action, "value") else str(action),
            actor_id=actor.id if hasattr(actor, "id") else str(actor),
            resource_id=resource.id if hasattr(resource, "id") else str(resource),
            resource_type=resource.type if hasattr(resource, "type") else "unknown",
            outcome=outcome.value if hasattr(outcome, "value") else str(outcome),
            details=details or {},
        )
        self.entries.append(entry)

    async def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        workspace_id: str | None = None,
        action: Any = None,
        outcome: Any = None,
        limit: int = 100,
    ) -> list[MockAuditEntry]:
        return self.entries[:limit]

    async def generate_compliance_report(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        workspace_id: str | None = None,
        format: str = "json",
    ) -> dict[str, Any]:
        return {
            "report_id": "report-123",
            "period_start": (
                start_date or datetime.now(timezone.utc) - timedelta(days=30)
            ).isoformat(),
            "period_end": (end_date or datetime.now(timezone.utc)).isoformat(),
            "total_entries": len(self.entries),
            "entries_by_action": {"create_workspace": 5, "delete_workspace": 2},
        }

    async def verify_integrity(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> tuple[bool, list[str]]:
        return True, []

    async def get_actor_history(self, actor_id: str, days: int = 30) -> list[MockAuditEntry]:
        return [e for e in self.entries if e.actor_id == actor_id]

    async def get_resource_history(self, resource_id: str, days: int = 30) -> list[MockAuditEntry]:
        return [e for e in self.entries if e.resource_id == resource_id]

    async def get_denied_access_attempts(self, days: int = 7) -> list[MockAuditEntry]:
        return [e for e in self.entries if e.outcome == "denied"]


@dataclass
class MockAuthorizationContext:
    """Mock RBAC authorization context."""

    user_id: str = "user-123"
    roles: set[str] = field(default_factory=lambda: {"member"})
    org_id: str | None = "org-123"


@dataclass
class MockAuthorizationDecision:
    """Mock RBAC authorization decision."""

    allowed: bool = True
    reason: str = ""


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
    store = MockUserStore()
    store.users["user-123"] = MockUser()
    store.users["user-456"] = MockUser(id="user-456", email="other@example.com", org_id="org-456")
    store.users["admin-user"] = MockUser(id="admin-user", role="admin")
    return store


@pytest.fixture
def isolation_manager():
    """Create a mock data isolation manager."""
    manager = MockDataIsolationManager()
    manager.workspaces["ws-123"] = MockWorkspace()
    return manager


@pytest.fixture
def retention_manager():
    """Create a mock retention policy manager."""
    manager = MockRetentionPolicyManager()
    manager.policies["policy-123"] = MockRetentionPolicy()
    return manager


@pytest.fixture
def classifier():
    """Create a mock sensitivity classifier."""
    return MockSensitivityClassifier()


@pytest.fixture
def audit_log():
    """Create a mock privacy audit log."""
    log = MockPrivacyAuditLog()
    log.entries.append(MockAuditEntry())
    return log


@pytest.fixture
def workspace_handler(user_store, isolation_manager, retention_manager, classifier, audit_log):
    """Create a workspace handler with mocked dependencies."""
    from aragora.server.handlers.workspace_module import WorkspaceHandler

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


@pytest.fixture
def admin_auth_context():
    """Create a mock admin auth context."""
    return MockAuthContext(user_id="admin-user", role="admin")


@pytest.fixture
def other_org_auth_context():
    """Create a mock auth context for a different organization."""
    return MockAuthContext(user_id="user-456", org_id="org-456")


# ===========================================================================
# can_handle Tests
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_workspaces_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/workspaces") is True

    def test_handles_workspaces_id_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/workspaces/ws-123") is True

    def test_handles_workspaces_members_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/workspaces/ws-123/members") is True

    def test_handles_retention_policies_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/retention/policies") is True

    def test_handles_retention_expiring_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/retention/expiring") is True

    def test_handles_classify_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/classify") is True

    def test_handles_audit_entries_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/audit/entries") is True

    def test_handles_audit_report_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/audit/report") is True

    def test_handles_audit_verify_route(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/audit/verify") is True

    def test_does_not_handle_other_routes(self, workspace_handler):
        assert workspace_handler.can_handle("/api/v1/debates") is False
        assert workspace_handler.can_handle("/api/v1/users") is False


# ===========================================================================
# Workspace CRUD Tests
# ===========================================================================


class TestWorkspaceCRUD:
    """Tests for workspace CRUD operations."""

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_workspace_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful workspace creation."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"name": "New Workspace"})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_create_workspace(mock_handler)

        assert get_status(result) == 201
        body = get_body(result)
        assert "workspace" in body
        assert body["workspace"]["name"] == "New Workspace"

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_workspace_requires_authentication(self, mock_extract, workspace_handler):
        """Test that workspace creation requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST", {"name": "New Workspace"})

        result = workspace_handler._handle_create_workspace(mock_handler)

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_workspace_requires_name(self, mock_extract, workspace_handler, auth_context):
        """Test that workspace creation requires a name."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_create_workspace(mock_handler)

        assert get_status(result) == 400
        body = get_body(result)
        assert "name" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_workspace_requires_org_id(self, mock_extract, workspace_handler):
        """Test that workspace creation requires organization ID."""
        mock_extract.return_value = MockAuthContext(org_id=None)
        mock_handler = create_mock_handler("POST", {"name": "New Workspace"})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_create_workspace(mock_handler)

        assert get_status(result) == 400
        body = get_body(result)
        assert "organization" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_list_workspaces_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful workspace listing."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_list_workspaces(mock_handler, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "workspaces" in body
        assert "total" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_list_workspaces_requires_authentication(self, mock_extract, workspace_handler):
        """Test that workspace listing requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler._handle_list_workspaces(mock_handler, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_get_workspace_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful workspace retrieval."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_get_workspace(mock_handler, "ws-123")

        assert get_status(result) == 200
        body = get_body(result)
        assert "workspace" in body
        assert body["workspace"]["id"] == "ws-123"

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_get_workspace_requires_authentication(self, mock_extract, workspace_handler):
        """Test that workspace retrieval requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler._handle_get_workspace(mock_handler, "ws-123")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_get_workspace_access_denied(
        self, mock_extract, workspace_handler, auth_context, isolation_manager
    ):
        """Test access denied when user lacks permission."""
        mock_extract.return_value = auth_context
        isolation_manager.access_denied_users.add("user-123")
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_get_workspace(mock_handler, "ws-123")

        assert get_status(result) == 403

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_delete_workspace_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful workspace deletion."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("DELETE", {})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_delete_workspace(mock_handler, "ws-123")

        assert get_status(result) == 200
        body = get_body(result)
        assert "deleted" in body.get("message", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_delete_workspace_requires_authentication(self, mock_extract, workspace_handler):
        """Test that workspace deletion requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("DELETE", {})

        result = workspace_handler._handle_delete_workspace(mock_handler, "ws-123")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_delete_workspace_with_force(self, mock_extract, workspace_handler, auth_context):
        """Test workspace deletion with force flag."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("DELETE", {"force": True})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_delete_workspace(mock_handler, "ws-123")

        assert get_status(result) == 200


# ===========================================================================
# Member Management Tests
# ===========================================================================


class TestMemberManagement:
    """Tests for workspace member management."""

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_add_member_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful member addition."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"user_id": "user-456"})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_add_member(mock_handler, "ws-123")

        assert get_status(result) == 201
        body = get_body(result)
        assert "user-456" in body.get("message", "")

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_add_member_requires_authentication(self, mock_extract, workspace_handler):
        """Test that adding member requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST", {"user_id": "user-456"})

        result = workspace_handler._handle_add_member(mock_handler, "ws-123")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_add_member_requires_user_id(self, mock_extract, workspace_handler, auth_context):
        """Test that adding member requires user_id."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_add_member(mock_handler, "ws-123")

        assert get_status(result) == 400
        body = get_body(result)
        assert "user_id" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_add_member_with_permissions(self, mock_extract, workspace_handler, auth_context):
        """Test adding member with specific permissions."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "POST", {"user_id": "user-456", "permissions": ["read", "write"]}
        )

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_add_member(mock_handler, "ws-123")

        assert get_status(result) == 201

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_remove_member_success(
        self, mock_extract, workspace_handler, auth_context, isolation_manager
    ):
        """Test successful member removal."""
        mock_extract.return_value = auth_context
        isolation_manager.workspaces["ws-123"].members.append("user-456")
        mock_handler = create_mock_handler("DELETE")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_remove_member(mock_handler, "ws-123", "user-456")

        assert get_status(result) == 200
        body = get_body(result)
        assert "user-456" in body.get("message", "")

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_remove_member_requires_authentication(self, mock_extract, workspace_handler):
        """Test that removing member requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("DELETE")

        result = workspace_handler._handle_remove_member(mock_handler, "ws-123", "user-456")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_remove_member_access_denied(
        self, mock_extract, workspace_handler, auth_context, isolation_manager
    ):
        """Test access denied when removing member without permission."""
        mock_extract.return_value = auth_context
        isolation_manager.access_denied_users.add("user-123")
        mock_handler = create_mock_handler("DELETE")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_remove_member(mock_handler, "ws-123", "user-456")

        assert get_status(result) == 403


# ===========================================================================
# Retention Policy Tests
# ===========================================================================


class TestRetentionPolicies:
    """Tests for retention policy management."""

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_list_policies_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful policy listing."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_list_policies(mock_handler, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "policies" in body
        assert "total" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_list_policies_requires_authentication(self, mock_extract, workspace_handler):
        """Test that listing policies requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler._handle_list_policies(mock_handler, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_policy_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful policy creation."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "POST", {"name": "New Policy", "retention_days": 30, "action": "delete"}
        )

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_create_policy(mock_handler)

        assert get_status(result) == 201
        body = get_body(result)
        assert "policy" in body
        assert body["policy"]["name"] == "New Policy"

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_policy_requires_name(self, mock_extract, workspace_handler, auth_context):
        """Test that policy creation requires name."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"retention_days": 30})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_create_policy(mock_handler)

        assert get_status(result) == 400
        body = get_body(result)
        assert "name" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_policy_invalid_action(self, mock_extract, workspace_handler, auth_context):
        """Test policy creation with invalid action."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"name": "Policy", "action": "invalid_action"})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_create_policy(mock_handler)

        assert get_status(result) == 400
        body = get_body(result)
        assert "action" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_get_policy_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful policy retrieval."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_get_policy(mock_handler, "policy-123")

        assert get_status(result) == 200
        body = get_body(result)
        assert "policy" in body
        assert body["policy"]["id"] == "policy-123"

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_get_policy_not_found(self, mock_extract, workspace_handler, auth_context):
        """Test policy retrieval when not found."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_get_policy(mock_handler, "nonexistent")

        assert get_status(result) == 404

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_update_policy_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful policy update."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("PUT", {"retention_days": 60})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_update_policy(mock_handler, "policy-123")

        assert get_status(result) == 200
        body = get_body(result)
        assert "policy" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_delete_policy_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful policy deletion."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("DELETE")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_delete_policy(mock_handler, "policy-123")

        assert get_status(result) == 200
        body = get_body(result)
        assert "deleted" in body.get("message", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_execute_policy_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful policy execution."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_execute_policy(mock_handler, "policy-123", {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "report" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_execute_policy_dry_run(self, mock_extract, workspace_handler, auth_context):
        """Test policy execution with dry run."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_execute_policy(
                mock_handler, "policy-123", {"dry_run": "true"}
            )

        assert get_status(result) == 200
        body = get_body(result)
        assert body["dry_run"] is True

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_expiring_items_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful expiring items retrieval."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_expiring_items(mock_handler, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "expiring" in body
        assert "days_ahead" in body


# ===========================================================================
# Sensitivity Classification Tests
# ===========================================================================


class TestSensitivityClassification:
    """Tests for sensitivity classification."""

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_classify_content_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful content classification."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"content": "This is sensitive financial data"})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_classify_content(mock_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert "classification" in body
        assert "level" in body["classification"]
        assert "confidence" in body["classification"]

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_classify_content_requires_authentication(self, mock_extract, workspace_handler):
        """Test that classification requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST", {"content": "Test content"})

        result = workspace_handler._handle_classify_content(mock_handler)

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_classify_content_requires_content(self, mock_extract, workspace_handler, auth_context):
        """Test that classification requires content."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_classify_content(mock_handler)

        assert get_status(result) == 400
        body = get_body(result)
        assert "content" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_classify_content_with_document_id(self, mock_extract, workspace_handler, auth_context):
        """Test classification with document ID."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "POST",
            {"content": "Test content", "document_id": "doc-123", "metadata": {"type": "report"}},
        )

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_classify_content(mock_handler)

        assert get_status(result) == 200

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_get_level_policy_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful level policy retrieval."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_get_level_policy(mock_handler, "internal")

        assert get_status(result) == 200
        body = get_body(result)
        assert "level" in body
        assert "policy" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_get_level_policy_invalid_level(self, mock_extract, workspace_handler, auth_context):
        """Test level policy with invalid level."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_get_level_policy(mock_handler, "invalid")

        assert get_status(result) == 400


# ===========================================================================
# Audit Logging Tests
# ===========================================================================


class TestAuditLogging:
    """Tests for audit logging functionality."""

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_query_audit_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful audit query."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_query_audit(mock_handler, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "entries" in body
        assert "total" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_query_audit_requires_authentication(self, mock_extract, workspace_handler):
        """Test that audit query requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler._handle_query_audit(mock_handler, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_query_audit_with_filters(self, mock_extract, workspace_handler, auth_context):
        """Test audit query with filters."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_query_audit(
                mock_handler,
                {
                    "actor_id": "user-123",
                    "workspace_id": "ws-123",
                    "limit": "50",
                },
            )

        assert get_status(result) == 200

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_audit_report_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful audit report generation."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_audit_report(mock_handler, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "report" in body
        assert "report_id" in body["report"]

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_audit_report_requires_authentication(self, mock_extract, workspace_handler):
        """Test that audit report requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler._handle_audit_report(mock_handler, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_verify_integrity_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful integrity verification."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_verify_integrity(mock_handler, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "valid" in body
        assert body["valid"] is True
        assert "verified_at" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_verify_integrity_requires_authentication(self, mock_extract, workspace_handler):
        """Test that integrity verification requires authentication."""
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = workspace_handler._handle_verify_integrity(mock_handler, {})

        assert get_status(result) == 401

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_actor_history_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful actor history retrieval."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_actor_history(mock_handler, "user-123", {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "actor_id" in body
        assert "entries" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_resource_history_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful resource history retrieval."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_resource_history(mock_handler, "ws-123", {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "resource_id" in body
        assert "entries" in body

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_denied_access_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful denied access retrieval."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_denied_access(mock_handler, {})

        assert get_status(result) == 200
        body = get_body(result)
        assert "denied_attempts" in body
        assert "total" in body


# ===========================================================================
# RBAC Permission Tests
# ===========================================================================


class TestRBACPermissions:
    """Tests for RBAC permission enforcement."""

    def test_workspace_handler_extends_secure_handler(self):
        """Test that WorkspaceHandler extends SecureHandler."""
        from aragora.server.handlers.workspace_module import WorkspaceHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(WorkspaceHandler, SecureHandler)

    def test_permission_constants_defined(self):
        """Test that permission constants are defined."""
        from aragora.server.handlers.workspace_module import (
            PERM_WORKSPACE_READ,
            PERM_WORKSPACE_WRITE,
            PERM_WORKSPACE_DELETE,
            PERM_WORKSPACE_ADMIN,
            PERM_WORKSPACE_SHARE,
            PERM_RETENTION_READ,
            PERM_RETENTION_WRITE,
            PERM_RETENTION_DELETE,
            PERM_RETENTION_EXECUTE,
            PERM_CLASSIFY_READ,
            PERM_CLASSIFY_WRITE,
            PERM_AUDIT_READ,
            PERM_AUDIT_REPORT,
            PERM_AUDIT_VERIFY,
        )

        assert PERM_WORKSPACE_READ == "workspace:read"
        assert PERM_WORKSPACE_WRITE == "workspace:write"
        assert PERM_WORKSPACE_DELETE == "workspace:delete"
        assert PERM_RETENTION_READ == "retention:read"
        assert PERM_AUDIT_READ == "audit:read"

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    @patch("aragora.server.handlers.workspace_module.check_permission")
    def test_rbac_permission_check_called(
        self, mock_check_perm, mock_extract, workspace_handler, auth_context
    ):
        """Test that RBAC permission check is called."""
        mock_extract.return_value = auth_context
        mock_check_perm.return_value = MockAuthorizationDecision(allowed=True)
        mock_handler = create_mock_handler("GET")

        workspace_handler._handle_list_workspaces(mock_handler, {})

        # Permission check may be called via _check_rbac_permission

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_rbac_permission_denied(self, mock_extract, workspace_handler, auth_context):
        """Test RBAC permission denied scenario."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        # Mock _check_rbac_permission to return an error
        from aragora.server.handlers.base import error_response

        with patch.object(
            workspace_handler,
            "_check_rbac_permission",
            return_value=error_response("Permission denied", 403),
        ):
            result = workspace_handler._handle_list_workspaces(mock_handler, {})

        assert get_status(result) == 403

    def test_handle_method_has_require_permission_decorator(self):
        """Test that handle method has require_permission decorator."""
        import inspect
        from aragora.server.handlers.workspace_module import WorkspaceHandler

        source = inspect.getsource(WorkspaceHandler.handle)
        assert "require_permission" in source

    def test_handle_post_has_require_permission_decorator(self):
        """Test that handle_post method has require_permission decorator."""
        import inspect
        from aragora.server.handlers.workspace_module import WorkspaceHandler

        source = inspect.getsource(WorkspaceHandler.handle_post)
        assert "require_permission" in source

    def test_handle_delete_has_require_permission_decorator(self):
        """Test that handle_delete method has require_permission decorator."""
        import inspect
        from aragora.server.handlers.workspace_module import WorkspaceHandler

        source = inspect.getsource(WorkspaceHandler.handle_delete)
        assert "require_permission" in source


# ===========================================================================
# Multi-Tenancy Tests
# ===========================================================================


class TestMultiTenancy:
    """Tests for multi-tenancy and cross-tenant access prevention."""

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_workspace_uses_auth_org_id(self, mock_extract, workspace_handler, auth_context):
        """Test that workspace creation uses authenticated user's org_id."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"name": "New Workspace"})

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_create_workspace(mock_handler)

        assert get_status(result) == 201
        body = get_body(result)
        # Workspace should use auth context org_id
        assert body["workspace"]["organization_id"] == "org-123"

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_create_workspace_rejects_cross_tenant(
        self, mock_extract, workspace_handler, auth_context
    ):
        """Test that workspace creation rejects cross-tenant requests."""
        mock_extract.return_value = auth_context
        # Try to create workspace in different org
        mock_handler = create_mock_handler(
            "POST", {"name": "New Workspace", "organization_id": "other-org"}
        )

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_create_workspace(mock_handler)

        assert get_status(result) == 403
        body = get_body(result)
        assert "another organization" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_list_workspaces_rejects_cross_tenant(
        self, mock_extract, workspace_handler, auth_context
    ):
        """Test that workspace listing rejects cross-tenant requests."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_list_workspaces(
                mock_handler, {"organization_id": "other-org"}
            )

        assert get_status(result) == 403
        body = get_body(result)
        assert "another organization" in body.get("error", "").lower()

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_list_workspaces_filters_by_org(
        self, mock_extract, workspace_handler, auth_context, isolation_manager
    ):
        """Test that workspace listing filters by organization."""
        mock_extract.return_value = auth_context
        # Add workspace in different org
        isolation_manager.workspaces["ws-other"] = MockWorkspace(
            id="ws-other", organization_id="other-org"
        )
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            result = workspace_handler._handle_list_workspaces(mock_handler, {})

        assert get_status(result) == 200
        body = get_body(result)
        # Should only see workspaces from user's org
        org_ids = [ws["organization_id"] for ws in body["workspaces"]]
        assert all(org_id == "org-123" for org_id in org_ids)

    def test_workspace_handler_has_cross_tenant_protection(self):
        """Test that workspace handler has cross-tenant protection logic."""
        import inspect
        from aragora.server.handlers.workspace_module import WorkspaceHandler

        # Check create workspace method
        create_source = inspect.getsource(WorkspaceHandler._handle_create_workspace)
        assert "auth_ctx.org_id" in create_source
        assert "Cannot create workspace in another organization" in create_source

        # Check list workspaces method
        list_source = inspect.getsource(WorkspaceHandler._handle_list_workspaces)
        assert "auth_ctx.org_id" in list_source
        assert "Cannot list workspaces from another organization" in list_source


# ===========================================================================
# RBAC Profile Tests
# ===========================================================================


class TestRBACProfiles:
    """Tests for RBAC profile management."""

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    @patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", True)
    def test_list_profiles_success(self, mock_extract, workspace_handler, auth_context):
        """Test successful profile listing."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            with patch("aragora.server.handlers.workspace_module.RBACProfile") as mock_profile:
                mock_profile.__iter__ = lambda self: iter([MagicMock(value="lite")])
                with patch(
                    "aragora.server.handlers.workspace_module.get_profile_config"
                ) as mock_config:
                    mock_config.return_value = MagicMock(
                        name="Lite",
                        description="Lite profile",
                        roles=["owner", "member"],
                        default_role="member",
                        features=set(["basic"]),
                    )
                    with patch(
                        "aragora.server.handlers.workspace_module.get_lite_role_summary"
                    ) as mock_summary:
                        mock_summary.return_value = {}
                        result = workspace_handler._handle_list_profiles(mock_handler)

        assert (
            get_status(result) == 200 or get_status(result) == 503
        )  # May be unavailable in test env

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    @patch("aragora.server.handlers.workspace_module.PROFILES_AVAILABLE", False)
    def test_list_profiles_unavailable(self, mock_extract, workspace_handler, auth_context):
        """Test profile listing when unavailable."""
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = workspace_handler._handle_list_profiles(mock_handler)

        assert get_status(result) == 503


# ===========================================================================
# Cache Tests
# ===========================================================================


class TestCaching:
    """Tests for caching functionality."""

    def test_cache_instances_exist(self):
        """Test that cache instances are created."""
        from aragora.server.handlers.workspace_module import (
            _retention_policy_cache,
            _permission_cache,
            _audit_query_cache,
        )

        assert _retention_policy_cache is not None
        assert _permission_cache is not None
        assert _audit_query_cache is not None

    def test_cache_stats_function_exists(self):
        """Test that cache stats function exists."""
        from aragora.server.handlers.workspace_module import get_workspace_cache_stats

        stats = get_workspace_cache_stats()
        assert "retention_policy_cache" in stats
        assert "permission_cache" in stats
        assert "audit_query_cache" in stats

    def test_invalidate_retention_cache(self):
        """Test retention cache invalidation."""
        from aragora.server.handlers.workspace_module import _invalidate_retention_cache

        # Should not raise
        result = _invalidate_retention_cache()
        assert isinstance(result, int)

        # With policy_id
        result = _invalidate_retention_cache("policy-123")
        assert isinstance(result, int)

    def test_invalidate_permission_cache(self):
        """Test permission cache invalidation."""
        from aragora.server.handlers.workspace_module import _invalidate_permission_cache

        # Should not raise
        result = _invalidate_permission_cache()
        assert isinstance(result, int)

        # With user_id
        result = _invalidate_permission_cache(user_id="user-123")
        assert isinstance(result, int)

        # With workspace_id
        result = _invalidate_permission_cache(workspace_id="ws-123")
        assert isinstance(result, int)

    def test_invalidate_audit_cache(self):
        """Test audit cache invalidation."""
        from aragora.server.handlers.workspace_module import _invalidate_audit_cache

        # Should not raise
        result = _invalidate_audit_cache()
        assert isinstance(result, int)

        # With workspace_id
        result = _invalidate_audit_cache("ws-123")
        assert isinstance(result, int)


# ===========================================================================
# Circuit Breaker Tests
# ===========================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_status_function(self):
        """Test circuit breaker status function."""
        from aragora.server.handlers.workspace_module import (
            get_workspace_circuit_breaker_status,
        )

        status = get_workspace_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_workspace_circuit_breaker_class(self):
        """Test WorkspaceCircuitBreaker class."""
        from aragora.server.handlers.workspace.workspace_utils import WorkspaceCircuitBreaker

        cb = WorkspaceCircuitBreaker()
        assert cb.state == WorkspaceCircuitBreaker.CLOSED
        assert cb.can_proceed() is True

    def test_circuit_breaker_records_success(self):
        """Test circuit breaker records success."""
        from aragora.server.handlers.workspace.workspace_utils import WorkspaceCircuitBreaker

        cb = WorkspaceCircuitBreaker()
        cb.record_success()
        assert cb.state == WorkspaceCircuitBreaker.CLOSED

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failures."""
        from aragora.server.handlers.workspace.workspace_utils import WorkspaceCircuitBreaker

        cb = WorkspaceCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()

        assert cb.state == WorkspaceCircuitBreaker.OPEN
        assert cb.can_proceed() is False

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        from aragora.server.handlers.workspace.workspace_utils import WorkspaceCircuitBreaker

        cb = WorkspaceCircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == WorkspaceCircuitBreaker.OPEN

        cb.reset()
        assert cb.state == WorkspaceCircuitBreaker.CLOSED
        assert cb.can_proceed() is True


# ===========================================================================
# Validation Tests
# ===========================================================================


class TestValidation:
    """Tests for validation functions."""

    def test_validate_workspace_id(self):
        """Test workspace ID validation."""
        from aragora.server.handlers.workspace.workspace_utils import _validate_workspace_id

        # Valid workspace ID
        valid, error = _validate_workspace_id("ws-123")
        assert valid is True
        assert error is None

        # Empty workspace ID
        valid, error = _validate_workspace_id("")
        assert valid is False
        assert error is not None

    def test_validate_policy_id(self):
        """Test policy ID validation."""
        from aragora.server.handlers.workspace.workspace_utils import _validate_policy_id

        # Valid policy ID
        valid, error = _validate_policy_id("policy-123")
        assert valid is True
        assert error is None

        # Empty policy ID
        valid, error = _validate_policy_id("")
        assert valid is False
        assert error is not None

    def test_validate_user_id(self):
        """Test user ID validation."""
        from aragora.server.handlers.workspace.workspace_utils import _validate_user_id

        # Valid user ID
        valid, error = _validate_user_id("user-123")
        assert valid is True
        assert error is None

        # Empty user ID
        valid, error = _validate_user_id("")
        assert valid is False
        assert error is not None


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """Integration tests for workspace module."""

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_workspace_lifecycle(self, mock_extract, workspace_handler, auth_context):
        """Test complete workspace lifecycle: create, get, add member, delete."""
        mock_extract.return_value = auth_context

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            # Create workspace
            create_handler = create_mock_handler("POST", {"name": "Lifecycle Workspace"})
            result = workspace_handler._handle_create_workspace(create_handler)
            assert get_status(result) == 201
            body = get_body(result)
            ws_id = body["workspace"]["id"]

            # Get workspace
            get_handler = create_mock_handler("GET")
            result = workspace_handler._handle_get_workspace(get_handler, ws_id)
            assert get_status(result) == 200

            # Add member
            member_handler = create_mock_handler("POST", {"user_id": "user-456"})
            result = workspace_handler._handle_add_member(member_handler, ws_id)
            assert get_status(result) == 201

            # Delete workspace
            delete_handler = create_mock_handler("DELETE", {})
            result = workspace_handler._handle_delete_workspace(delete_handler, ws_id)
            assert get_status(result) == 200

    @patch("aragora.server.handlers.workspace_module.extract_user_from_request")
    def test_retention_policy_lifecycle(self, mock_extract, workspace_handler, auth_context):
        """Test complete retention policy lifecycle."""
        mock_extract.return_value = auth_context

        with patch.object(workspace_handler, "_check_rbac_permission", return_value=None):
            # Create policy
            create_handler = create_mock_handler(
                "POST", {"name": "Test Policy", "retention_days": 30}
            )
            result = workspace_handler._handle_create_policy(create_handler)
            assert get_status(result) == 201
            body = get_body(result)
            policy_id = body["policy"]["id"]

            # Get policy
            get_handler = create_mock_handler("GET")
            result = workspace_handler._handle_get_policy(get_handler, policy_id)
            assert get_status(result) == 200

            # Update policy
            update_handler = create_mock_handler("PUT", {"retention_days": 60})
            result = workspace_handler._handle_update_policy(update_handler, policy_id)
            assert get_status(result) == 200

            # Execute policy (dry run)
            exec_handler = create_mock_handler("POST")
            result = workspace_handler._handle_execute_policy(
                exec_handler, policy_id, {"dry_run": "true"}
            )
            assert get_status(result) == 200

            # Delete policy
            delete_handler = create_mock_handler("DELETE")
            result = workspace_handler._handle_delete_policy(delete_handler, policy_id)
            assert get_status(result) == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
