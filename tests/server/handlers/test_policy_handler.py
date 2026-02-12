"""
Tests for PolicyHandler - Policy and Compliance HTTP endpoints.

Tests cover:
- Route matching (can_handle)
- RBAC permission enforcement
- Input validation
- Happy path operations
- Error handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import os

import pytest

from aragora.server.handlers.policy import PolicyHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockPolicyRule:
    """Mock policy rule for testing."""

    rule_id: str = "rule-001"
    name: str = "Test Rule"
    description: str = "A test rule"
    severity: str = "medium"
    enabled: bool = True
    custom_threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity,
            "enabled": self.enabled,
            "custom_threshold": self.custom_threshold,
            "metadata": self.metadata,
        }


@dataclass
class MockPolicy:
    """Mock policy for testing."""

    id: str = "pol_abc123"
    name: str = "Test Policy"
    description: str = "A test compliance policy"
    framework_id: str = "soc2"
    workspace_id: str = "default"
    vertical_id: str = "finance"
    level: str = "recommended"
    enabled: bool = True
    rules: list[MockPolicyRule] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "rules_count": len(self.rules),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata,
        }


@dataclass
class MockViolation:
    """Mock violation for testing."""

    id: str = "viol_xyz789"
    policy_id: str = "pol_abc123"
    rule_id: str = "rule-001"
    rule_name: str = "Test Rule"
    framework_id: str = "soc2"
    vertical_id: str = "finance"
    workspace_id: str = "default"
    severity: str = "high"
    status: str = "open"
    description: str = "Test violation detected"
    source: str = "manual_check"
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    resolution_notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
            "metadata": self.metadata,
        }


@dataclass
class MockComplianceResult:
    """Mock compliance check result."""

    compliant: bool = True
    score: float = 95.0
    issues: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.compliant,
            "score": self.score,
            "issues": self.issues,
        }


class MockPolicyStore:
    """Mock policy store for testing."""

    def __init__(self):
        self._policies: dict[str, MockPolicy] = {}
        self._violations: dict[str, MockViolation] = {}

    def list_policies(
        self,
        workspace_id: str | None = None,
        vertical_id: str | None = None,
        framework_id: str | None = None,
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

    def get_policy(self, policy_id: str) -> MockPolicy | None:
        return self._policies.get(policy_id)

    def create_policy(self, policy: MockPolicy) -> MockPolicy:
        self._policies[policy.id] = policy
        return policy

    def update_policy(
        self, policy_id: str, data: dict[str, Any], changed_by: str | None = None
    ) -> MockPolicy | None:
        if policy_id not in self._policies:
            return None
        policy = self._policies[policy_id]
        for key, value in data.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        return policy

    def delete_policy(self, policy_id: str) -> bool:
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False

    def toggle_policy(
        self, policy_id: str, enabled: bool, changed_by: str | None = None
    ) -> bool:
        if policy_id not in self._policies:
            return False
        self._policies[policy_id].enabled = enabled
        return True

    def list_violations(
        self,
        policy_id: str | None = None,
        workspace_id: str | None = None,
        vertical_id: str | None = None,
        framework_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MockViolation]:
        violations = list(self._violations.values())
        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]
        if workspace_id:
            violations = [v for v in violations if v.workspace_id == workspace_id]
        if status:
            violations = [v for v in violations if v.status == status]
        if severity:
            violations = [v for v in violations if v.severity == severity]
        return violations[offset : offset + limit]

    def get_violation(self, violation_id: str) -> MockViolation | None:
        return self._violations.get(violation_id)

    def create_violation(self, violation: MockViolation) -> MockViolation:
        self._violations[violation.id] = violation
        return violation

    def update_violation_status(
        self,
        violation_id: str,
        status: str,
        resolved_by: str | None = None,
        resolution_notes: str | None = None,
    ) -> MockViolation | None:
        if violation_id not in self._violations:
            return None
        violation = self._violations[violation_id]
        violation.status = status
        violation.resolved_by = resolved_by
        violation.resolution_notes = resolution_notes
        if status == "resolved":
            violation.resolved_at = datetime.now(timezone.utc)
        return violation

    def count_violations(
        self, workspace_id: str | None = None, status: str | None = None
    ) -> dict[str, int]:
        violations = list(self._violations.values())
        if workspace_id:
            violations = [v for v in violations if v.workspace_id == workspace_id]
        if status:
            violations = [v for v in violations if v.status == status]
        return {
            "total": len(violations),
            "critical": len([v for v in violations if v.severity == "critical"]),
            "high": len([v for v in violations if v.severity == "high"]),
            "medium": len([v for v in violations if v.severity == "medium"]),
            "low": len([v for v in violations if v.severity == "low"]),
        }


class MockComplianceManager:
    """Mock compliance framework manager for testing."""

    def check(self, content: str, frameworks=None, min_severity=None) -> MockComplianceResult:
        return MockComplianceResult(compliant=True, score=95.0, issues=[])


def create_mock_handler(
    method: str = "GET",
    body: dict[str, Any] | None = None,
    path: str = "/api/v1/policies",
) -> MagicMock:
    """Create a mock HTTP handler for testing."""
    mock = MagicMock()
    mock.command = method
    mock.path = path

    if body is not None:
        body_bytes = json.dumps(body).encode()
    else:
        body_bytes = b"{}"

    mock.rfile = MagicMock()
    mock.rfile.read = MagicMock(return_value=body_bytes)

    mock.headers = {"Content-Length": str(len(body_bytes))}
    mock.client_address = ("127.0.0.1", 12345)
    mock.user_context = MagicMock()
    mock.user_context.user_id = "test_user"

    return mock


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def mock_policy_store():
    """Create mock policy store with sample data."""
    store = MockPolicyStore()
    store._policies["pol_abc123"] = MockPolicy()
    store._policies["pol_disabled"] = MockPolicy(
        id="pol_disabled",
        name="Disabled Policy",
        enabled=False,
    )
    store._violations["viol_xyz789"] = MockViolation()
    return store


@pytest.fixture
def handler(mock_server_context, mock_policy_store):
    """Create handler with mocked dependencies."""
    h = PolicyHandler(mock_server_context)
    h._get_policy_store = MagicMock(return_value=mock_policy_store)
    h._get_compliance_manager = MagicMock(return_value=MockComplianceManager())
    return h


# ===========================================================================
# Route Matching Tests
# ===========================================================================


class TestPolicyHandlerRouting:
    """Test request routing."""

    def test_can_handle_policies_list_path(self, handler):
        """Test that handler recognizes policy list path."""
        assert handler.can_handle("/api/v1/policies", "GET")
        assert handler.can_handle("/api/v1/policies", "POST")

    def test_can_handle_policy_detail_paths(self, handler):
        """Test that handler recognizes policy detail paths."""
        assert handler.can_handle("/api/v1/policies/pol_abc123", "GET")
        assert handler.can_handle("/api/v1/policies/pol_abc123", "PATCH")
        assert handler.can_handle("/api/v1/policies/pol_abc123", "DELETE")
        assert handler.can_handle("/api/v1/policies/pol_abc123/toggle", "POST")
        assert handler.can_handle("/api/v1/policies/pol_abc123/violations", "GET")

    def test_can_handle_compliance_paths(self, handler):
        """Test that handler recognizes compliance paths."""
        assert handler.can_handle("/api/v1/compliance/violations", "GET")
        assert handler.can_handle("/api/v1/compliance/violations/viol_xyz789", "GET")
        assert handler.can_handle("/api/v1/compliance/violations/viol_xyz789", "PATCH")
        assert handler.can_handle("/api/v1/compliance/check", "POST")
        assert handler.can_handle("/api/v1/compliance/stats", "GET")

    def test_cannot_handle_other_paths(self, handler):
        """Test that handler rejects non-policy paths."""
        assert not handler.can_handle("/api/v1/backups", "GET")
        assert not handler.can_handle("/api/v1/debates", "GET")
        assert not handler.can_handle("/api/v2/policies", "GET")
        assert not handler.can_handle("/api/policies", "GET")


# ===========================================================================
# RBAC Permission Tests
# ===========================================================================


class TestPolicyHandlerRBAC:
    """Test RBAC permission enforcement."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_list_policies_requires_policies_read(
        self, mock_server_context, mock_policy_store
    ):
        """Test that listing policies requires policies:read permission."""
        # Enable real auth testing
        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = PolicyHandler(mock_server_context)
            h._get_policy_store = MagicMock(return_value=mock_policy_store)
            mock_handler = create_mock_handler()

            # Without proper auth context, should return 401
            result = await h.handle("/api/v1/policies", "GET", mock_handler)
            assert result is not None
            assert result.status_code == 401
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_create_policy_requires_policies_create(
        self, mock_server_context, mock_policy_store
    ):
        """Test that creating a policy requires policies:create permission."""
        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = PolicyHandler(mock_server_context)
            h._get_policy_store = MagicMock(return_value=mock_policy_store)
            mock_handler = create_mock_handler(
                method="POST",
                body={"name": "New Policy", "framework_id": "soc2", "vertical_id": "finance"},
            )

            result = await h.handle("/api/v1/policies", "POST", mock_handler)
            assert result is not None
            assert result.status_code == 401
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_delete_policy_requires_policies_delete(
        self, mock_server_context, mock_policy_store
    ):
        """Test that deleting a policy requires policies:delete permission."""
        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = PolicyHandler(mock_server_context)
            h._get_policy_store = MagicMock(return_value=mock_policy_store)
            mock_handler = create_mock_handler(method="DELETE")

            result = await h.handle("/api/v1/policies/pol_abc123", "DELETE", mock_handler)
            assert result is not None
            assert result.status_code == 401
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_toggle_policy_requires_policies_update(
        self, mock_server_context, mock_policy_store
    ):
        """Test that toggling a policy requires policies:update permission."""
        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = PolicyHandler(mock_server_context)
            h._get_policy_store = MagicMock(return_value=mock_policy_store)
            mock_handler = create_mock_handler(
                method="POST",
                body={"enabled": False},
            )

            result = await h.handle("/api/v1/policies/pol_abc123/toggle", "POST", mock_handler)
            assert result is not None
            assert result.status_code == 401
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_compliance_check_requires_policies_create(
        self, mock_server_context, mock_policy_store
    ):
        """Test that compliance check requires policies:create permission."""
        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = PolicyHandler(mock_server_context)
            h._get_compliance_manager = MagicMock(return_value=MockComplianceManager())
            mock_handler = create_mock_handler(
                method="POST",
                body={"content": "Test content to check"},
            )

            result = await h.handle("/api/v1/compliance/check", "POST", mock_handler)
            assert result is not None
            assert result.status_code == 401
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]


# ===========================================================================
# Input Validation Tests
# ===========================================================================


class TestPolicyHandlerValidation:
    """Test input validation."""

    @pytest.mark.asyncio
    async def test_create_policy_missing_name(self, handler):
        """Test creating policy without name returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"framework_id": "soc2", "vertical_id": "finance"},
        )

        result = await handler.handle("/api/v1/policies", "POST", mock_handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "name" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_policy_missing_framework_id(self, handler):
        """Test creating policy without framework_id returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"name": "Test Policy", "vertical_id": "finance"},
        )

        result = await handler.handle("/api/v1/policies", "POST", mock_handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "framework_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_policy_missing_vertical_id(self, handler):
        """Test creating policy without vertical_id returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"name": "Test Policy", "framework_id": "soc2"},
        )

        result = await handler.handle("/api/v1/policies", "POST", mock_handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "vertical_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_policy_invalid_json(self, handler):
        """Test creating policy with invalid JSON returns 400."""
        mock_handler = create_mock_handler(method="POST")
        mock_handler.rfile.read.return_value = b"not valid json"

        result = await handler.handle("/api/v1/policies", "POST", mock_handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "json" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_update_violation_invalid_status(self, handler):
        """Test updating violation with invalid status returns 400."""
        mock_handler = create_mock_handler(
            method="PATCH",
            body={"status": "invalid_status"},
        )

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_xyz789", "PATCH", mock_handler
        )
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "status" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_update_violation_missing_status(self, handler):
        """Test updating violation without status returns 400."""
        mock_handler = create_mock_handler(
            method="PATCH",
            body={"resolution_notes": "Fixed the issue"},
        )

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_xyz789", "PATCH", mock_handler
        )
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "status" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_compliance_check_missing_content(self, handler):
        """Test compliance check without content returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"frameworks": ["soc2"]},
        )

        result = await handler.handle("/api/v1/compliance/check", "POST", mock_handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "content" in body.get("error", "").lower()


# ===========================================================================
# Happy Path Tests
# ===========================================================================


class TestListPolicies:
    """Test list policies endpoint."""

    @pytest.mark.asyncio
    async def test_list_policies_success(self, handler):
        """Test listing policies returns correct format."""
        mock_handler = create_mock_handler()

        result = await handler.handle("/api/v1/policies", "GET", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "policies" in body
        assert "total" in body
        assert isinstance(body["policies"], list)

    @pytest.mark.asyncio
    async def test_list_policies_with_filters(self, handler):
        """Test listing policies with query filters."""
        mock_handler = create_mock_handler(
            path="/api/v1/policies?enabled_only=true&framework_id=soc2"
        )

        result = await handler.handle("/api/v1/policies", "GET", mock_handler)
        assert result.status_code == 200


class TestGetPolicy:
    """Test get single policy endpoint."""

    @pytest.mark.asyncio
    async def test_get_policy_success(self, handler):
        """Test getting a specific policy."""
        mock_handler = create_mock_handler()

        result = await handler.handle("/api/v1/policies/pol_abc123", "GET", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "policy" in body
        assert body["policy"]["id"] == "pol_abc123"

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, handler):
        """Test getting non-existent policy returns 404."""
        mock_handler = create_mock_handler()

        result = await handler.handle("/api/v1/policies/nonexistent", "GET", mock_handler)
        assert result.status_code == 404


class TestCreatePolicy:
    """Test create policy endpoint."""

    @pytest.mark.asyncio
    async def test_create_policy_success(self, handler):
        """Test creating a new policy."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "name": "New Policy",
                "description": "A new compliance policy",
                "framework_id": "gdpr",
                "vertical_id": "healthcare",
                "level": "mandatory",
            },
        )

        result = await handler.handle("/api/v1/policies", "POST", mock_handler)
        assert result.status_code == 201
        body = json.loads(result.body)
        assert "policy" in body
        assert body["policy"]["name"] == "New Policy"
        assert "message" in body

    @pytest.mark.asyncio
    async def test_create_policy_with_rules(self, handler):
        """Test creating a policy with rules."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "name": "Policy With Rules",
                "framework_id": "soc2",
                "vertical_id": "finance",
                "rules": [
                    {
                        "rule_id": "r1",
                        "name": "Rule 1",
                        "description": "Test rule",
                        "severity": "high",
                    }
                ],
            },
        )

        result = await handler.handle("/api/v1/policies", "POST", mock_handler)
        assert result.status_code == 201


class TestTogglePolicy:
    """Test toggle policy endpoint."""

    @pytest.mark.asyncio
    async def test_toggle_policy_enable(self, handler, mock_policy_store):
        """Test enabling a policy."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"enabled": True},
        )

        result = await handler.handle("/api/v1/policies/pol_disabled/toggle", "POST", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_toggle_policy_disable(self, handler):
        """Test disabling a policy."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"enabled": False},
        )

        result = await handler.handle("/api/v1/policies/pol_abc123/toggle", "POST", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_toggle_policy_not_found(self, handler):
        """Test toggling non-existent policy returns 404."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"enabled": True},
        )

        result = await handler.handle("/api/v1/policies/nonexistent/toggle", "POST", mock_handler)
        assert result.status_code == 404


class TestComplianceCheck:
    """Test compliance check endpoint."""

    @pytest.mark.asyncio
    async def test_compliance_check_success(self, handler):
        """Test running a compliance check."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "content": "This is test content to check for compliance.",
                "min_severity": "low",
            },
        )

        result = await handler.handle("/api/v1/compliance/check", "POST", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "result" in body
        assert "compliant" in body
        assert "score" in body


class TestComplianceStats:
    """Test compliance statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler):
        """Test getting compliance statistics."""
        mock_handler = create_mock_handler()

        result = await handler.handle("/api/v1/compliance/stats", "GET", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "policies" in body
        assert "violations" in body
        assert "risk_score" in body


class TestViolations:
    """Test violation endpoints."""

    @pytest.mark.asyncio
    async def test_list_violations_success(self, handler):
        """Test listing all violations."""
        mock_handler = create_mock_handler()

        result = await handler.handle("/api/v1/compliance/violations", "GET", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violations" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_get_violation_success(self, handler):
        """Test getting a specific violation."""
        mock_handler = create_mock_handler()

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_xyz789", "GET", mock_handler
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violation" in body
        assert body["violation"]["id"] == "viol_xyz789"

    @pytest.mark.asyncio
    async def test_get_violation_not_found(self, handler):
        """Test getting non-existent violation returns 404."""
        mock_handler = create_mock_handler()

        result = await handler.handle(
            "/api/v1/compliance/violations/nonexistent", "GET", mock_handler
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_update_violation_status(self, handler):
        """Test updating violation status."""
        mock_handler = create_mock_handler(
            method="PATCH",
            body={
                "status": "resolved",
                "resolution_notes": "Fixed the issue",
            },
        )

        result = await handler.handle(
            "/api/v1/compliance/violations/viol_xyz789", "PATCH", mock_handler
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violation" in body
        assert body["violation"]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_get_policy_violations(self, handler):
        """Test getting violations for a specific policy."""
        mock_handler = create_mock_handler()

        result = await handler.handle("/api/v1/policies/pol_abc123/violations", "GET", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violations" in body
        assert body["policy_id"] == "pol_abc123"


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestPolicyHandlerErrors:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_policy_store_unavailable(self, mock_server_context):
        """Test handling when policy store is unavailable."""
        h = PolicyHandler(mock_server_context)
        h._get_policy_store = MagicMock(return_value=None)
        mock_handler = create_mock_handler()

        result = await h.handle("/api/v1/policies", "GET", mock_handler)
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_compliance_manager_unavailable(self, mock_server_context):
        """Test handling when compliance manager is unavailable."""
        h = PolicyHandler(mock_server_context)
        h._get_compliance_manager = MagicMock(return_value=None)
        mock_handler = create_mock_handler(
            method="POST",
            body={"content": "Test content"},
        )

        result = await h.handle("/api/v1/compliance/check", "POST", mock_handler)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_delete_policy_not_found(self, handler):
        """Test deleting non-existent policy returns 404."""
        mock_handler = create_mock_handler(method="DELETE")

        result = await handler.handle("/api/v1/policies/nonexistent", "DELETE", mock_handler)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_update_policy_not_found(self, handler):
        """Test updating non-existent policy returns 404."""
        mock_handler = create_mock_handler(
            method="PATCH",
            body={"name": "Updated Name"},
        )

        result = await handler.handle("/api/v1/policies/nonexistent", "PATCH", mock_handler)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_update_policy_empty_body(self, handler):
        """Test updating policy with empty body returns 400."""
        mock_handler = create_mock_handler(
            method="PATCH",
            body={},
        )

        result = await handler.handle("/api/v1/policies/pol_abc123", "PATCH", mock_handler)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_policy_violations_for_nonexistent_policy(self, handler):
        """Test getting violations for non-existent policy returns 404."""
        mock_handler = create_mock_handler()

        result = await handler.handle(
            "/api/v1/policies/nonexistent/violations", "GET", mock_handler
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_error_response_format(self, handler):
        """Test that error responses have correct format."""
        mock_handler = create_mock_handler()

        result = await handler.handle("/api/v1/policies/nonexistent", "GET", mock_handler)
        assert result.status_code == 404
        body = json.loads(result.body)
        assert "error" in body
        assert isinstance(body["error"], str)


# ===========================================================================
# Update and Delete Tests
# ===========================================================================


class TestUpdatePolicy:
    """Test update policy endpoint."""

    @pytest.mark.asyncio
    async def test_update_policy_success(self, handler):
        """Test updating a policy."""
        mock_handler = create_mock_handler(
            method="PATCH",
            body={"name": "Updated Policy Name", "description": "Updated description"},
        )

        result = await handler.handle("/api/v1/policies/pol_abc123", "PATCH", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "policy" in body
        assert "message" in body


class TestDeletePolicy:
    """Test delete policy endpoint."""

    @pytest.mark.asyncio
    async def test_delete_policy_success(self, handler, mock_policy_store):
        """Test deleting a policy."""
        mock_handler = create_mock_handler(method="DELETE")

        result = await handler.handle("/api/v1/policies/pol_abc123", "DELETE", mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "message" in body
        assert body["policy_id"] == "pol_abc123"


# ===========================================================================
# Path Validation Tests
# ===========================================================================


class TestPathValidation:
    """Test path segment validation."""

    @pytest.mark.asyncio
    async def test_invalid_policy_id_special_characters(self, handler):
        """Test that policy IDs with special characters are rejected."""
        mock_handler = create_mock_handler()

        # IDs with special characters should fail validation
        # SAFE_ID_PATTERN only allows [a-zA-Z0-9_-]{1,64}
        result = await handler.handle("/api/v1/policies/pol$abc!@#", "GET", mock_handler)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_violation_id_special_characters(self, handler):
        """Test that violation IDs with special characters are rejected."""
        mock_handler = create_mock_handler()

        # IDs with dots and slashes should fail validation
        result = await handler.handle(
            "/api/v1/compliance/violations/viol.id/test", "GET", mock_handler
        )
        # With extra path segments, this won't match any route (returns None)
        # But with a valid path format and invalid ID chars, it should return 400
        assert result is None  # Route doesn't match with extra segments

    @pytest.mark.asyncio
    async def test_policy_id_too_long(self, handler):
        """Test that overly long policy IDs are rejected."""
        mock_handler = create_mock_handler()

        # ID exceeding 64 characters should fail
        long_id = "a" * 100
        result = await handler.handle(f"/api/v1/policies/{long_id}", "GET", mock_handler)
        assert result.status_code == 400
