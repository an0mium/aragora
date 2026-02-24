"""
Tests for FastAPI compliance route endpoints.

Covers:
- GET  /api/v2/compliance/status             - Compliance framework status
- GET  /api/v2/compliance/policies           - List policies
- POST /api/v2/compliance/artifacts/generate - Generate compliance artifacts
- GET  /api/v2/compliance/audit-log          - Query audit log
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def mock_compliance_framework():
    """Create a mock compliance framework."""
    fw = MagicMock()
    fw.get_status = MagicMock(return_value={
        "overall_status": "partially_compliant",
        "controls": [
            {
                "control_id": "CC-1.1",
                "name": "Control Environment",
                "description": "Organization demonstrates commitment to integrity",
                "status": "passing",
                "evidence_count": 5,
                "last_assessed": "2026-02-15T10:00:00",
            },
            {
                "control_id": "CC-1.2",
                "name": "Board Oversight",
                "description": "Board exercises oversight",
                "status": "not_assessed",
                "evidence_count": 0,
                "last_assessed": None,
            },
            {
                "control_id": "CC-2.1",
                "name": "Risk Assessment",
                "description": "Organization identifies risks",
                "status": "failing",
                "evidence_count": 2,
                "last_assessed": "2026-02-14T10:00:00",
            },
        ],
        "last_assessment": "2026-02-15T10:00:00",
    })
    fw.list_policies = MagicMock(return_value=[])
    fw.count_policies = MagicMock(return_value=0)
    fw.generate_artifact = MagicMock(return_value={
        "framework": "soc2",
        "status": "generated",
        "integrity_hash": "sha256_abc123",
    })
    return fw


@pytest.fixture
def mock_audit_store():
    """Create a mock audit store."""
    store = MagicMock()
    store.query = MagicMock(return_value=[])
    store.count = MagicMock(return_value=0)
    return store


@pytest.fixture
def client(app, mock_compliance_framework, mock_audit_store):
    """Create a test client with mocked context."""
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
        "compliance_framework": mock_compliance_framework,
        "audit_store": mock_audit_store,
    }
    return TestClient(app, raise_server_exceptions=False)


def _override_auth(client, permissions=None):
    """Override auth for write operations."""
    from aragora.server.fastapi.dependencies.auth import require_authenticated
    from aragora.rbac.models import AuthorizationContext

    if permissions is None:
        permissions = {"compliance:write", "audit:read"}

    auth_ctx = AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id="ws-1",
        roles={"admin"},
        permissions=permissions,
    )
    client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx


# =============================================================================
# GET /api/v2/compliance/status
# =============================================================================


class TestComplianceStatus:
    """Tests for GET /api/v2/compliance/status."""

    def test_status_returns_200(self, client):
        """Status returns compliance overview."""
        response = client.get("/api/v2/compliance/status")
        assert response.status_code == 200
        data = response.json()
        assert data["framework"] == "soc2"
        assert data["overall_status"] == "partially_compliant"
        assert data["controls_total"] == 3
        assert data["controls_passing"] == 1
        assert data["controls_failing"] == 1
        assert data["controls_not_assessed"] == 1

    def test_status_coverage_percent(self, client):
        """Status computes coverage percentage."""
        response = client.get("/api/v2/compliance/status")
        assert response.status_code == 200
        data = response.json()
        # 1 passing out of 3 controls = 33.3%
        assert data["coverage_percent"] == pytest.approx(33.3, abs=0.1)

    def test_status_with_framework_param(self, client, mock_compliance_framework):
        """Status passes framework parameter."""
        response = client.get("/api/v2/compliance/status?framework=gdpr")
        assert response.status_code == 200
        mock_compliance_framework.get_status.assert_called_once_with(framework="gdpr")

    def test_status_includes_controls(self, client):
        """Status response includes control details."""
        response = client.get("/api/v2/compliance/status")
        assert response.status_code == 200
        data = response.json()
        assert len(data["controls"]) == 3
        ctrl = data["controls"][0]
        assert ctrl["control_id"] == "CC-1.1"
        assert ctrl["name"] == "Control Environment"
        assert ctrl["status"] == "passing"
        assert ctrl["evidence_count"] == 5

    def test_status_when_framework_unavailable(self, app):
        """Status returns defaults when framework is not configured."""
        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "compliance_framework": None,
        }
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/v2/compliance/status")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "not_configured"
        assert data["controls_total"] == 0


# =============================================================================
# GET /api/v2/compliance/policies
# =============================================================================


class TestListPolicies:
    """Tests for GET /api/v2/compliance/policies."""

    def test_list_returns_200_empty(self, client):
        """List policies returns 200 with empty list."""
        response = client.get("/api/v2/compliance/policies")
        assert response.status_code == 200
        data = response.json()
        assert data["policies"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_list_with_data(self, client, mock_compliance_framework):
        """List returns policy summaries."""
        mock_compliance_framework.list_policies.return_value = [
            {
                "id": "pol-001",
                "name": "Data Retention Policy",
                "description": "Defines data retention periods",
                "framework": "gdpr",
                "status": "active",
                "enforcement": "enforced",
                "created_at": "2026-01-01T00:00:00",
            },
            {
                "id": "pol-002",
                "name": "Access Control Policy",
                "description": "Defines RBAC rules",
                "framework": "soc2",
                "status": "active",
                "enforcement": "enforced",
                "created_at": "2026-01-15T00:00:00",
            },
        ]
        mock_compliance_framework.count_policies.return_value = 2

        response = client.get("/api/v2/compliance/policies")
        assert response.status_code == 200
        data = response.json()
        assert len(data["policies"]) == 2
        assert data["total"] == 2
        assert data["policies"][0]["name"] == "Data Retention Policy"
        assert data["policies"][1]["framework"] == "soc2"

    def test_list_with_framework_filter(self, client, mock_compliance_framework):
        """List passes framework filter."""
        response = client.get("/api/v2/compliance/policies?framework=gdpr")
        assert response.status_code == 200
        mock_compliance_framework.list_policies.assert_called_once_with(
            limit=50, offset=0, framework="gdpr", status=None,
        )

    def test_list_with_status_filter(self, client, mock_compliance_framework):
        """List passes status filter."""
        response = client.get("/api/v2/compliance/policies?status=active")
        assert response.status_code == 200
        mock_compliance_framework.list_policies.assert_called_once_with(
            limit=50, offset=0, framework=None, status="active",
        )

    def test_list_pagination(self, client, mock_compliance_framework):
        """List passes pagination params."""
        response = client.get("/api/v2/compliance/policies?limit=10&offset=5")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 5

    def test_list_limit_validation(self, client):
        """List limit must be between 1 and 100."""
        response = client.get("/api/v2/compliance/policies?limit=0")
        assert response.status_code == 422

        response = client.get("/api/v2/compliance/policies?limit=101")
        assert response.status_code == 422

    def test_list_when_framework_unavailable(self, app):
        """List returns empty when framework is unavailable."""
        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "compliance_framework": None,
        }
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/v2/compliance/policies")
        assert response.status_code == 200
        data = response.json()
        assert data["policies"] == []
        assert data["total"] == 0


# =============================================================================
# POST /api/v2/compliance/artifacts/generate
# =============================================================================


class TestGenerateArtifact:
    """Tests for POST /api/v2/compliance/artifacts/generate."""

    def test_generate_returns_200(self, client, mock_compliance_framework):
        """Generate returns artifact with content."""
        _override_auth(client)

        response = client.post(
            "/api/v2/compliance/artifacts/generate",
            json={
                "framework": "soc2",
                "artifact_type": "full_bundle",
                "debate_ids": ["debate-001"],
            },
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["artifact_id"].startswith("ca_")
        assert data["framework"] == "soc2"
        assert data["artifact_type"] == "full_bundle"
        assert "content" in data

    def test_generate_requires_framework(self, client):
        """Generate without framework returns 422."""
        _override_auth(client)

        response = client.post(
            "/api/v2/compliance/artifacts/generate",
            json={"artifact_type": "full_bundle"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 422

    def test_generate_requires_auth(self, client):
        """Generate without auth returns 401."""
        response = client.post(
            "/api/v2/compliance/artifacts/generate",
            json={"framework": "soc2"},
        )
        assert response.status_code == 401

    def test_generate_calls_framework(self, client, mock_compliance_framework):
        """Generate calls the compliance framework generate method."""
        _override_auth(client)

        response = client.post(
            "/api/v2/compliance/artifacts/generate",
            json={
                "framework": "soc2",
                "artifact_type": "technical_doc",
                "debate_ids": ["d1", "d2"],
            },
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        mock_compliance_framework.generate_artifact.assert_called_once()

    def test_generate_fallback_when_no_engine(self, app):
        """Generate returns basic artifact when no compliance engine."""
        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "compliance_framework": None,
        }
        client = TestClient(app, raise_server_exceptions=False)
        _override_auth(client)

        response = client.post(
            "/api/v2/compliance/artifacts/generate",
            json={"framework": "soc2"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "note" in data["content"]


# =============================================================================
# GET /api/v2/compliance/audit-log
# =============================================================================


class TestAuditLog:
    """Tests for GET /api/v2/compliance/audit-log."""

    def test_audit_log_returns_200_empty(self, client):
        """Audit log returns 200 with empty list."""
        _override_auth(client)

        response = client.get("/api/v2/compliance/audit-log")
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["entries"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_audit_log_with_entries(self, client, mock_audit_store):
        """Audit log returns entries from store."""
        _override_auth(client)

        mock_audit_store.query.return_value = [
            {
                "id": "aud-001",
                "timestamp": "2026-02-15T10:00:00",
                "action": "debate.created",
                "actor": "user-1",
                "resource_type": "debate",
                "resource_id": "debate-001",
                "details": {"task": "Security review"},
            },
            {
                "id": "aud-002",
                "timestamp": "2026-02-15T10:05:00",
                "action": "receipt.generated",
                "actor": "system",
                "resource_type": "receipt",
                "resource_id": "rcpt-001",
                "details": {},
            },
        ]
        mock_audit_store.count.return_value = 2

        response = client.get("/api/v2/compliance/audit-log")
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert len(data["entries"]) == 2
        assert data["total"] == 2
        assert data["entries"][0]["action"] == "debate.created"
        assert data["entries"][0]["actor"] == "user-1"

    def test_audit_log_with_action_filter(self, client, mock_audit_store):
        """Audit log passes action filter."""
        _override_auth(client)

        response = client.get("/api/v2/compliance/audit-log?action=debate.created")
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        mock_audit_store.query.assert_called_once_with(
            limit=50, offset=0, action="debate.created",
        )

    def test_audit_log_with_actor_filter(self, client, mock_audit_store):
        """Audit log passes actor filter."""
        _override_auth(client)

        response = client.get("/api/v2/compliance/audit-log?actor=user-1")
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        mock_audit_store.query.assert_called_once_with(
            limit=50, offset=0, actor="user-1",
        )

    def test_audit_log_with_resource_type_filter(self, client, mock_audit_store):
        """Audit log passes resource_type filter."""
        _override_auth(client)

        response = client.get("/api/v2/compliance/audit-log?resource_type=debate")
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        mock_audit_store.query.assert_called_once_with(
            limit=50, offset=0, resource_type="debate",
        )

    def test_audit_log_pagination(self, client, mock_audit_store):
        """Audit log passes pagination params."""
        _override_auth(client)

        response = client.get("/api/v2/compliance/audit-log?limit=10&offset=20")
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 20

    def test_audit_log_requires_auth(self, client):
        """Audit log without auth returns 401."""
        response = client.get("/api/v2/compliance/audit-log")
        assert response.status_code == 401

    def test_audit_log_limit_validation(self, client):
        """Audit log limit must be between 1 and 500."""
        _override_auth(client)

        response = client.get("/api/v2/compliance/audit-log?limit=0")
        client.app.dependency_overrides.clear()

        assert response.status_code == 422

    def test_audit_log_when_store_unavailable(self, app):
        """Audit log returns empty when store is unavailable."""
        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "audit_store": None,
        }
        client = TestClient(app, raise_server_exceptions=False)
        _override_auth(client)

        response = client.get("/api/v2/compliance/audit-log")
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["entries"] == []
        assert data["total"] == 0
