"""
Tests for PolicyHandler - Policy and Compliance endpoint handlers.

Tests cover:
- Policy CRUD operations (list, get, create, update, delete)
- Policy toggle enabled/disabled status
- Policy violations retrieval
- Compliance violations list and detail
- Compliance check endpoint
- Compliance statistics
- Error handling and validation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.policy import PolicyHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockPolicyRule:
    """Mock policy rule for testing."""

    id: str = "rule_001"
    name: str = "Test Rule"
    description: str = "A test rule"
    pattern: str = ".*"
    severity: str = "medium"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "severity": self.severity,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockPolicyRule":
        return cls(
            id=data.get("id", "rule_001"),
            name=data.get("name", "Test Rule"),
            description=data.get("description", ""),
            pattern=data.get("pattern", ".*"),
            severity=data.get("severity", "medium"),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MockPolicy:
    """Mock policy for testing."""

    id: str = "pol_test123"
    name: str = "Test Policy"
    description: str = "A test policy"
    framework_id: str = "framework_001"
    workspace_id: str = "default"
    vertical_id: str = "vertical_001"
    level: str = "recommended"
    enabled: bool = True
    rules: list[MockPolicyRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "framework_id": self.framework_id,
            "workspace_id": self.workspace_id,
            "vertical_id": self.vertical_id,
            "level": self.level,
            "enabled": self.enabled,
            "rules": [r.to_dict() for r in self.rules],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class MockViolation:
    """Mock violation for testing."""

    id: str = "viol_test123"
    policy_id: str = "pol_test123"
    rule_id: str = "rule_001"
    rule_name: str = "Test Rule"
    framework_id: str = "framework_001"
    vertical_id: str = "vertical_001"
    workspace_id: str = "default"
    severity: str = "medium"
    status: str = "open"
    description: str = "Test violation"
    source: str = "manual_check"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "framework_id": self.framework_id,
            "vertical_id": self.vertical_id,
            "workspace_id": self.workspace_id,
            "severity": self.severity,
            "status": self.status,
            "description": self.description,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class MockComplianceIssue:
    """Mock compliance issue from check."""

    rule_id: str = "rule_001"
    description: str = "Compliance issue found"
    framework: str = "framework_001"
    severity: Any = None  # Will be set to mock enum
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockComplianceResult:
    """Mock compliance check result."""

    compliant: bool = True
    score: float = 0.95
    issues: list[MockComplianceIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.compliant,
            "score": self.score,
            "issues": [
                {
                    "rule_id": i.rule_id,
                    "description": i.description,
                    "framework": i.framework,
                    "severity": i.severity.value if hasattr(i.severity, "value") else i.severity,
                    "metadata": i.metadata,
                }
                for i in self.issues
            ],
        }


class MockPolicyStore:
    """Mock policy store for testing."""

    def __init__(self):
        self._policies: dict[str, MockPolicy] = {}
        self._violations: dict[str, MockViolation] = {}

    def list_policies(
        self,
        workspace_id: Optional[str] = None,
        vertical_id: Optional[str] = None,
        framework_id: Optional[str] = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MockPolicy]:
        policies = list(self._policies.values())
        if workspace_id:
            policies = [p for p in policies if p.workspace_id == workspace_id]
        if vertical_id:
            policies = [p for p in policies if p.vertical_id == vertical_id]
        if framework_id:
            policies = [p for p in policies if p.framework_id == framework_id]
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        return policies[offset : offset + limit]

    def get_policy(self, policy_id: str) -> Optional[MockPolicy]:
        return self._policies.get(policy_id)

    def create_policy(self, policy: MockPolicy) -> MockPolicy:
        self._policies[policy.id] = policy
        return policy

    def update_policy(
        self, policy_id: str, data: dict[str, Any], changed_by: Optional[str] = None
    ) -> Optional[MockPolicy]:
        policy = self._policies.get(policy_id)
        if policy is None:
            return None
        for key, value in data.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        policy.updated_at = datetime.now(timezone.utc)
        return policy

    def delete_policy(self, policy_id: str) -> bool:
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False

    def toggle_policy(
        self, policy_id: str, enabled: bool, changed_by: Optional[str] = None
    ) -> bool:
        policy = self._policies.get(policy_id)
        if policy is None:
            return False
        policy.enabled = enabled
        return True

    def list_violations(
        self,
        policy_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        vertical_id: Optional[str] = None,
        framework_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MockViolation]:
        violations = list(self._violations.values())
        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]
        if workspace_id:
            violations = [v for v in violations if v.workspace_id == workspace_id]
        if vertical_id:
            violations = [v for v in violations if v.vertical_id == vertical_id]
        if framework_id:
            violations = [v for v in violations if v.framework_id == framework_id]
        if status:
            violations = [v for v in violations if v.status == status]
        if severity:
            violations = [v for v in violations if v.severity == severity]
        return violations[offset : offset + limit]

    def get_violation(self, violation_id: str) -> Optional[MockViolation]:
        return self._violations.get(violation_id)

    def create_violation(self, violation: MockViolation) -> MockViolation:
        self._violations[violation.id] = violation
        return violation

    def update_violation_status(
        self,
        violation_id: str,
        status: str,
        resolved_by: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> Optional[MockViolation]:
        violation = self._violations.get(violation_id)
        if violation is None:
            return None
        violation.status = status
        if status == "resolved":
            violation.resolved_at = datetime.now(timezone.utc)
        violation.resolved_by = resolved_by
        violation.resolution_notes = resolution_notes
        return violation

    def count_violations(
        self, workspace_id: Optional[str] = None, status: Optional[str] = None
    ) -> dict[str, int]:
        violations = list(self._violations.values())
        if workspace_id:
            violations = [v for v in violations if v.workspace_id == workspace_id]
        if status:
            violations = [v for v in violations if v.status == status]

        counts = {"total": len(violations), "critical": 0, "high": 0, "medium": 0, "low": 0}
        for v in violations:
            if v.severity in counts:
                counts[v.severity] += 1
        return counts


class MockComplianceFrameworkManager:
    """Mock compliance framework manager for testing."""

    def __init__(self):
        self.check_result = MockComplianceResult()

    def check(
        self,
        content: str,
        frameworks: Optional[list[str]] = None,
        min_severity: Any = None,
    ) -> MockComplianceResult:
        return self.check_result


def create_mock_http_handler(
    body: Optional[dict[str, Any]] = None,
    path: str = "/api/v1/policies",
    user_id: Optional[str] = None,
) -> MagicMock:
    """Create a mock HTTP handler with request body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body_bytes = json.dumps(body).encode("utf-8") if body else b""
    handler.headers = {
        "Content-Length": str(len(body_bytes)),
        "Content-Type": "application/json",
    }
    handler.rfile = BytesIO(body_bytes)
    handler.path = path

    if user_id:
        handler.user_context = MagicMock()
        handler.user_context.user_id = user_id
    else:
        handler.user_context = None

    return handler


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def mock_policy_store():
    """Create mock policy store with sample data."""
    store = MockPolicyStore()
    store._policies["pol_test123"] = MockPolicy()
    store._policies["pol_disabled"] = MockPolicy(
        id="pol_disabled",
        name="Disabled Policy",
        enabled=False,
    )
    store._violations["viol_test123"] = MockViolation()
    store._violations["viol_critical"] = MockViolation(
        id="viol_critical",
        severity="critical",
        status="open",
    )
    return store


@pytest.fixture
def mock_compliance_manager():
    """Create mock compliance manager."""
    return MockComplianceFrameworkManager()


@pytest.fixture
def handler(mock_server_context, mock_policy_store, mock_compliance_manager):
    """Create handler with mocked dependencies."""
    h = PolicyHandler(mock_server_context)

    # Patch the internal methods to return mocks
    h._get_policy_store = MagicMock(return_value=mock_policy_store)
    h._get_compliance_manager = MagicMock(return_value=mock_compliance_manager)

    return h


# ===========================================================================
# Handler Routing Tests
# ===========================================================================


class TestPolicyHandlerRouting:
    """Test request routing for PolicyHandler."""

    def test_can_handle_policies_path(self, handler):
        """Test that handler recognizes policy paths."""
        assert handler.can_handle("/api/v1/policies", "GET")
        assert handler.can_handle("/api/v1/policies", "POST")
        assert handler.can_handle("/api/v1/policies/pol_test123", "GET")
        assert handler.can_handle("/api/v1/policies/pol_test123", "PATCH")
        assert handler.can_handle("/api/v1/policies/pol_test123", "DELETE")
        assert handler.can_handle("/api/v1/policies/pol_test123/toggle", "POST")
        assert handler.can_handle("/api/v1/policies/pol_test123/violations", "GET")

    def test_can_handle_compliance_path(self, handler):
        """Test that handler recognizes compliance paths."""
        assert handler.can_handle("/api/v1/compliance/violations", "GET")
        assert handler.can_handle("/api/v1/compliance/violations/viol_123", "GET")
        assert handler.can_handle("/api/v1/compliance/violations/viol_123", "PATCH")
        assert handler.can_handle("/api/v1/compliance/check", "POST")
        assert handler.can_handle("/api/v1/compliance/stats", "GET")

    def test_cannot_handle_other_paths(self, handler):
        """Test that handler rejects non-policy paths."""
        assert not handler.can_handle("/api/v1/debates", "GET")
        assert not handler.can_handle("/api/v1/agents", "GET")
        assert not handler.can_handle("/api/v2/policies", "GET")


# ===========================================================================
# Policy List Tests
# ===========================================================================


class TestListPolicies:
    """Test list policies endpoint."""

    @pytest.mark.asyncio
    async def test_list_policies_success(self, handler, mock_policy_store):
        """Test listing policies returns correct format."""
        http_handler = create_mock_http_handler(path="/api/v1/policies")

        result = await handler.handle("/api/v1/policies", "GET", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "policies" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    @pytest.mark.asyncio
    async def test_list_policies_with_filters(self, handler, mock_policy_store):
        """Test listing policies with query filters."""
        http_handler = create_mock_http_handler(
            path="/api/v1/policies?workspace_id=default&enabled_only=true"
        )

        result = await handler.handle("/api/v1/policies", "GET", http_handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_policies_store_unavailable(self, handler):
        """Test list policies returns 503 when store unavailable."""
        handler._get_policy_store = MagicMock(return_value=None)
        http_handler = create_mock_http_handler()

        result = await handler.handle("/api/v1/policies", "GET", http_handler)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Policy Get Tests
# ===========================================================================


class TestGetPolicy:
    """Test get single policy endpoint."""

    @pytest.mark.asyncio
    async def test_get_policy_success(self, handler, mock_policy_store):
        """Test getting a specific policy."""
        http_handler = create_mock_http_handler(path="/api/v1/policies/pol_test123")

        result = await handler.handle("/api/v1/policies/pol_test123", "GET", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "policy" in body
        assert body["policy"]["id"] == "pol_test123"

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, handler):
        """Test getting non-existent policy returns 404."""
        http_handler = create_mock_http_handler(path="/api/v1/policies/nonexistent")

        result = await handler.handle("/api/v1/policies/nonexistent", "GET", http_handler)

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_policy_invalid_id(self, handler):
        """Test getting policy with invalid ID returns 400."""
        # Use angle brackets which are rejected by SAFE_ID_PATTERN
        http_handler = create_mock_http_handler(path="/api/v1/policies/pol<invalid>")

        result = await handler.handle("/api/v1/policies/pol<invalid>", "GET", http_handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Policy Create Tests
# ===========================================================================


class TestCreatePolicy:
    """Test create policy endpoint."""

    @pytest.mark.asyncio
    async def test_create_policy_success(self, handler, mock_policy_store):
        """Test creating a new policy."""
        http_handler = create_mock_http_handler(
            body={
                "name": "New Policy",
                "framework_id": "framework_001",
                "vertical_id": "vertical_001",
                "description": "A new test policy",
            },
            path="/api/v1/policies",
            user_id="test_user",
        )

        with patch("aragora.server.handlers.policy.audit_security") as mock_audit:
            result = await handler.handle("/api/v1/policies", "POST", http_handler)

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert "policy" in body
        assert body["policy"]["name"] == "New Policy"
        assert "message" in body
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_policy_missing_required_fields(self, handler):
        """Test creating policy without required fields fails."""
        http_handler = create_mock_http_handler(
            body={"name": "Incomplete Policy"},
            path="/api/v1/policies",
        )

        result = await handler.handle("/api/v1/policies", "POST", http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_create_policy_invalid_json(self, handler):
        """Test creating policy with invalid JSON fails."""
        http_handler = MagicMock()
        http_handler.headers = {"Content-Length": "10"}
        http_handler.rfile = BytesIO(b"not json!!")

        result = await handler.handle("/api/v1/policies", "POST", http_handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_policy_store_unavailable(self, handler):
        """Test create policy returns 503 when store unavailable."""
        handler._get_policy_store = MagicMock(return_value=None)
        http_handler = create_mock_http_handler(
            body={
                "name": "New Policy",
                "framework_id": "framework_001",
                "vertical_id": "vertical_001",
            },
            path="/api/v1/policies",
        )

        result = await handler.handle("/api/v1/policies", "POST", http_handler)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Policy Update Tests
# ===========================================================================


class TestUpdatePolicy:
    """Test update policy endpoint."""

    @pytest.mark.asyncio
    async def test_update_policy_success(self, handler, mock_policy_store):
        """Test updating a policy."""
        http_handler = create_mock_http_handler(
            body={"name": "Updated Policy Name", "description": "Updated description"},
            path="/api/v1/policies/pol_test123",
            user_id="test_user",
        )

        with patch("aragora.server.handlers.policy.audit_security") as mock_audit:
            result = await handler.handle("/api/v1/policies/pol_test123", "PATCH", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "policy" in body
        assert "message" in body
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_policy_not_found(self, handler):
        """Test updating non-existent policy returns 404."""
        http_handler = create_mock_http_handler(
            body={"name": "Updated Name"},
            path="/api/v1/policies/nonexistent",
        )

        result = await handler.handle("/api/v1/policies/nonexistent", "PATCH", http_handler)

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_update_policy_no_data(self, handler):
        """Test updating policy with empty body fails."""
        http_handler = create_mock_http_handler(
            body={},
            path="/api/v1/policies/pol_test123",
        )

        result = await handler.handle("/api/v1/policies/pol_test123", "PATCH", http_handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Policy Delete Tests
# ===========================================================================


class TestDeletePolicy:
    """Test delete policy endpoint."""

    @pytest.mark.asyncio
    async def test_delete_policy_success(self, handler, mock_policy_store):
        """Test deleting a policy."""
        http_handler = create_mock_http_handler(path="/api/v1/policies/pol_test123")

        with patch("aragora.server.handlers.policy.audit_security") as mock_audit:
            result = await handler.handle("/api/v1/policies/pol_test123", "DELETE", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "message" in body
        assert body["policy_id"] == "pol_test123"
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_policy_not_found(self, handler):
        """Test deleting non-existent policy returns 404."""
        http_handler = create_mock_http_handler(path="/api/v1/policies/nonexistent")

        result = await handler.handle("/api/v1/policies/nonexistent", "DELETE", http_handler)

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Policy Toggle Tests
# ===========================================================================


class TestTogglePolicy:
    """Test toggle policy enabled status endpoint."""

    @pytest.mark.asyncio
    async def test_toggle_policy_enable(self, handler, mock_policy_store):
        """Test enabling a disabled policy."""
        http_handler = create_mock_http_handler(
            body={"enabled": True},
            path="/api/v1/policies/pol_disabled/toggle",
            user_id="test_user",
        )

        with patch("aragora.server.handlers.policy.audit_security") as mock_audit:
            result = await handler.handle(
                "/api/v1/policies/pol_disabled/toggle", "POST", http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["enabled"] is True
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_toggle_policy_disable(self, handler, mock_policy_store):
        """Test disabling an enabled policy."""
        http_handler = create_mock_http_handler(
            body={"enabled": False},
            path="/api/v1/policies/pol_test123/toggle",
        )

        result = await handler.handle("/api/v1/policies/pol_test123/toggle", "POST", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_toggle_policy_auto_toggle(self, handler, mock_policy_store):
        """Test toggling policy without explicit enabled value."""
        http_handler = create_mock_http_handler(
            body={},
            path="/api/v1/policies/pol_test123/toggle",
        )

        result = await handler.handle("/api/v1/policies/pol_test123/toggle", "POST", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should toggle from True to False
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_toggle_policy_not_found(self, handler):
        """Test toggling non-existent policy returns 404."""
        http_handler = create_mock_http_handler(
            body={"enabled": True},
            path="/api/v1/policies/nonexistent/toggle",
        )

        result = await handler.handle("/api/v1/policies/nonexistent/toggle", "POST", http_handler)

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Policy Violations Tests
# ===========================================================================


class TestGetPolicyViolations:
    """Test get violations for specific policy endpoint."""

    @pytest.mark.asyncio
    async def test_get_policy_violations_success(self, handler, mock_policy_store):
        """Test getting violations for a policy."""
        # Add a violation for the policy
        mock_policy_store._violations["viol_for_policy"] = MockViolation(
            id="viol_for_policy",
            policy_id="pol_test123",
        )

        http_handler = create_mock_http_handler(path="/api/v1/policies/pol_test123/violations")

        result = await handler.handle(
            "/api/v1/policies/pol_test123/violations", "GET", http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violations" in body
        assert body["policy_id"] == "pol_test123"

    @pytest.mark.asyncio
    async def test_get_policy_violations_policy_not_found(self, handler):
        """Test getting violations for non-existent policy returns 404."""
        http_handler = create_mock_http_handler(path="/api/v1/policies/nonexistent/violations")

        result = await handler.handle(
            "/api/v1/policies/nonexistent/violations", "GET", http_handler
        )

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Compliance Violations List Tests
# ===========================================================================


class TestListViolations:
    """Test list all violations endpoint."""

    @pytest.mark.asyncio
    async def test_list_violations_success(self, handler, mock_policy_store):
        """Test listing all violations."""
        http_handler = create_mock_http_handler(path="/api/v1/compliance/violations")

        result = await handler.handle("/api/v1/compliance/violations", "GET", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violations" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_list_violations_with_filters(self, handler, mock_policy_store):
        """Test listing violations with filters."""
        http_handler = create_mock_http_handler(
            path="/api/v1/compliance/violations?status=open&severity=critical"
        )

        result = await handler.handle("/api/v1/compliance/violations", "GET", http_handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Compliance Violation Detail Tests
# ===========================================================================


class TestGetViolation:
    """Test get single violation endpoint."""

    @pytest.mark.asyncio
    async def test_get_violation_success(self, handler, mock_policy_store):
        """Test getting a specific violation."""
        http_handler = create_mock_http_handler(path="/api/v1/compliance/violations/viol_test123")

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_test123", "GET", http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violation" in body
        assert body["violation"]["id"] == "viol_test123"

    @pytest.mark.asyncio
    async def test_get_violation_not_found(self, handler):
        """Test getting non-existent violation returns 404."""
        http_handler = create_mock_http_handler(path="/api/v1/compliance/violations/nonexistent")

        result = await handler.handle(
            "/api/v1/compliance/violations/nonexistent", "GET", http_handler
        )

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Update Violation Status Tests
# ===========================================================================


class TestUpdateViolationStatus:
    """Test update violation status endpoint."""

    @pytest.mark.asyncio
    async def test_update_violation_resolved(self, handler, mock_policy_store):
        """Test resolving a violation."""
        http_handler = create_mock_http_handler(
            body={"status": "resolved", "resolution_notes": "Issue fixed"},
            path="/api/v1/compliance/violations/viol_test123",
            user_id="test_user",
        )

        with patch("aragora.server.handlers.policy.audit_security") as mock_audit:
            result = await handler.handle(
                "/api/v1/compliance/violations/viol_test123", "PATCH", http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violation" in body
        assert body["violation"]["status"] == "resolved"
        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_violation_investigating(self, handler, mock_policy_store):
        """Test marking violation as investigating."""
        http_handler = create_mock_http_handler(
            body={"status": "investigating"},
            path="/api/v1/compliance/violations/viol_test123",
        )

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_test123", "PATCH", http_handler
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_update_violation_false_positive(self, handler, mock_policy_store):
        """Test marking violation as false positive."""
        http_handler = create_mock_http_handler(
            body={"status": "false_positive"},
            path="/api/v1/compliance/violations/viol_test123",
        )

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_test123", "PATCH", http_handler
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_update_violation_invalid_status(self, handler):
        """Test updating violation with invalid status fails."""
        http_handler = create_mock_http_handler(
            body={"status": "invalid_status"},
            path="/api/v1/compliance/violations/viol_test123",
        )

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_test123", "PATCH", http_handler
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_update_violation_missing_status(self, handler):
        """Test updating violation without status fails."""
        http_handler = create_mock_http_handler(
            body={"resolution_notes": "Notes only"},
            path="/api/v1/compliance/violations/viol_test123",
        )

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_test123", "PATCH", http_handler
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_update_violation_not_found(self, handler):
        """Test updating non-existent violation returns 404."""
        http_handler = create_mock_http_handler(
            body={"status": "resolved"},
            path="/api/v1/compliance/violations/nonexistent",
        )

        result = await handler.handle(
            "/api/v1/compliance/violations/nonexistent", "PATCH", http_handler
        )

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Compliance Check Tests
# ===========================================================================


class TestComplianceCheck:
    """Test compliance check endpoint."""

    @pytest.mark.asyncio
    async def test_compliance_check_success(self, handler, mock_compliance_manager):
        """Test running a compliance check."""
        http_handler = create_mock_http_handler(
            body={"content": "Test content to check for compliance"},
            path="/api/v1/compliance/check",
        )

        result = await handler.handle("/api/v1/compliance/check", "POST", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "result" in body
        assert "compliant" in body
        assert "score" in body

    @pytest.mark.asyncio
    async def test_compliance_check_with_frameworks(self, handler, mock_compliance_manager):
        """Test compliance check with specific frameworks."""
        http_handler = create_mock_http_handler(
            body={
                "content": "Test content",
                "frameworks": ["framework_001", "framework_002"],
            },
            path="/api/v1/compliance/check",
        )

        result = await handler.handle("/api/v1/compliance/check", "POST", http_handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_compliance_check_missing_content(self, handler):
        """Test compliance check without content fails."""
        http_handler = create_mock_http_handler(
            body={"frameworks": ["framework_001"]},
            path="/api/v1/compliance/check",
        )

        result = await handler.handle("/api/v1/compliance/check", "POST", http_handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_compliance_check_invalid_severity(self, handler):
        """Test compliance check with invalid min_severity fails."""
        http_handler = create_mock_http_handler(
            body={"content": "Test content", "min_severity": "invalid"},
            path="/api/v1/compliance/check",
        )

        result = await handler.handle("/api/v1/compliance/check", "POST", http_handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_compliance_check_manager_unavailable(self, handler):
        """Test compliance check returns 503 when manager unavailable."""
        handler._get_compliance_manager = MagicMock(return_value=None)
        http_handler = create_mock_http_handler(
            body={"content": "Test content"},
            path="/api/v1/compliance/check",
        )

        result = await handler.handle("/api/v1/compliance/check", "POST", http_handler)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Compliance Stats Tests
# ===========================================================================


class TestComplianceStats:
    """Test compliance statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler, mock_policy_store):
        """Test getting compliance statistics."""
        http_handler = create_mock_http_handler(path="/api/v1/compliance/stats")

        result = await handler.handle("/api/v1/compliance/stats", "GET", http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "policies" in body
        assert "violations" in body
        assert "risk_score" in body

    @pytest.mark.asyncio
    async def test_get_stats_with_workspace(self, handler, mock_policy_store):
        """Test getting stats filtered by workspace."""
        http_handler = create_mock_http_handler(
            path="/api/v1/compliance/stats?workspace_id=default"
        )

        result = await handler.handle("/api/v1/compliance/stats", "GET", http_handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_stats_store_unavailable(self, handler):
        """Test get stats returns 503 when store unavailable."""
        handler._get_policy_store = MagicMock(return_value=None)
        http_handler = create_mock_http_handler(path="/api/v1/compliance/stats")

        result = await handler.handle("/api/v1/compliance/stats", "GET", http_handler)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in policy operations."""

    @pytest.mark.asyncio
    async def test_list_policies_exception(self, handler, mock_policy_store):
        """Test list policies handles exceptions."""
        mock_policy_store.list_policies = MagicMock(side_effect=Exception("Database error"))
        http_handler = create_mock_http_handler(path="/api/v1/policies")

        result = await handler.handle("/api/v1/policies", "GET", http_handler)

        assert result is not None
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_get_policy_exception(self, handler, mock_policy_store):
        """Test get policy handles exceptions."""
        mock_policy_store.get_policy = MagicMock(side_effect=Exception("Database error"))
        http_handler = create_mock_http_handler(path="/api/v1/policies/pol_test123")

        result = await handler.handle("/api/v1/policies/pol_test123", "GET", http_handler)

        assert result is not None
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_create_policy_exception(self, handler, mock_policy_store):
        """Test create policy handles exceptions."""
        mock_policy_store.create_policy = MagicMock(side_effect=Exception("Database error"))
        http_handler = create_mock_http_handler(
            body={
                "name": "Test Policy",
                "framework_id": "framework_001",
                "vertical_id": "vertical_001",
            },
            path="/api/v1/policies",
        )

        result = await handler.handle("/api/v1/policies", "POST", http_handler)

        assert result is not None
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_compliance_check_exception(self, handler, mock_compliance_manager):
        """Test compliance check handles exceptions."""
        mock_compliance_manager.check = MagicMock(side_effect=Exception("Check failed"))
        http_handler = create_mock_http_handler(
            body={"content": "Test content"},
            path="/api/v1/compliance/check",
        )

        result = await handler.handle("/api/v1/compliance/check", "POST", http_handler)

        assert result is not None
        assert result.status_code == 500


# ===========================================================================
# Path Segment Validation Tests
# ===========================================================================


class TestPathValidation:
    """Tests for path segment validation."""

    @pytest.mark.asyncio
    async def test_policy_id_with_special_chars(self, handler):
        """Test policy ID with special characters is rejected."""
        http_handler = create_mock_http_handler(path="/api/v1/policies/pol<script>")

        result = await handler.handle("/api/v1/policies/pol<script>", "GET", http_handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_violation_id_with_special_chars(self, handler):
        """Test violation ID with special characters is rejected."""
        # Use angle brackets which are rejected by SAFE_ID_PATTERN
        http_handler = create_mock_http_handler(path="/api/v1/compliance/violations/viol<script>")

        result = await handler.handle(
            "/api/v1/compliance/violations/viol<script>", "GET", http_handler
        )

        assert result is not None
        assert result.status_code == 400
