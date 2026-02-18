"""Tests for coordination handler endpoints."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.coordination.cross_workspace import (
    CrossWorkspaceCoordinator,
    CrossWorkspaceRequest,
    CrossWorkspaceResult,
    DataSharingConsent,
    FederatedWorkspace,
    FederationMode,
    FederationPolicy,
    OperationType,
    SharingScope,
)
from aragora.server.handlers.control_plane import ControlPlaneHandler


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def coordinator() -> CrossWorkspaceCoordinator:
    """Create a fresh coordinator for tests."""
    return CrossWorkspaceCoordinator()


@pytest.fixture
def handler(coordinator: CrossWorkspaceCoordinator) -> ControlPlaneHandler:
    """Create a handler with a coordination coordinator."""
    ctx: dict[str, Any] = {
        "coordination_coordinator": coordinator,
        "control_plane_coordinator": MagicMock(),
    }
    h = ControlPlaneHandler(ctx)
    return h


@pytest.fixture
def mock_http_handler() -> MagicMock:
    """Create a mock HTTP handler for POST requests."""
    m = MagicMock()
    m.path = "/api/v1/coordination/workspaces"
    return m


def _set_body(mock_handler: MagicMock, body: dict[str, Any]) -> None:
    """Set up mock handler to return JSON body."""
    raw = json.dumps(body).encode()
    mock_handler.rfile.read.return_value = raw
    mock_handler.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}


# ============================================================================
# Workspace Endpoints
# ============================================================================


class TestRegisterWorkspace:
    def test_register_workspace_success(self, handler: ControlPlaneHandler, mock_http_handler: MagicMock):
        _set_body(mock_http_handler, {
            "id": "ws-1",
            "name": "Primary",
            "org_id": "org-1",
            "federation_mode": "readonly",
        })
        result = handler._handle_register_workspace({
            "id": "ws-1",
            "name": "Primary",
            "org_id": "org-1",
            "federation_mode": "readonly",
        })
        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["id"] == "ws-1"
        assert data["name"] == "Primary"
        assert data["federation_mode"] == "readonly"

    def test_register_workspace_missing_id(self, handler: ControlPlaneHandler):
        result = handler._handle_register_workspace({"name": "No ID"})
        assert result.status_code == 400
        assert "id" in json.loads(result.body).get("error", "").lower()

    def test_register_workspace_invalid_mode(self, handler: ControlPlaneHandler):
        result = handler._handle_register_workspace({
            "id": "ws-bad",
            "federation_mode": "invalid_mode",
        })
        assert result.status_code == 400


class TestListWorkspaces:
    def test_list_empty(self, handler: ControlPlaneHandler):
        result = handler._handle_list_workspaces({})
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 0
        assert data["workspaces"] == []

    def test_list_after_register(self, handler: ControlPlaneHandler, coordinator: CrossWorkspaceCoordinator):
        ws = FederatedWorkspace(id="ws-1", name="Test", org_id="org-1")
        coordinator.register_workspace(ws)
        result = handler._handle_list_workspaces({})
        data = json.loads(result.body)
        assert data["total"] == 1
        assert data["workspaces"][0]["id"] == "ws-1"


class TestUnregisterWorkspace:
    def test_unregister_success(self, handler: ControlPlaneHandler, coordinator: CrossWorkspaceCoordinator):
        ws = FederatedWorkspace(id="ws-1", name="Test", org_id="org-1")
        coordinator.register_workspace(ws)
        result = handler._handle_unregister_workspace("ws-1")
        assert result.status_code == 200
        assert json.loads(result.body)["unregistered"] is True
        # Verify removed
        assert len(coordinator.list_workspaces()) == 0


# ============================================================================
# Federation Policy Endpoints
# ============================================================================


class TestCreateFederationPolicy:
    def test_create_policy_success(self, handler: ControlPlaneHandler):
        result = handler._handle_create_federation_policy({
            "name": "test-policy",
            "description": "A test policy",
            "mode": "bidirectional",
            "sharing_scope": "metadata",
            "allowed_operations": ["read_knowledge", "query_mound"],
        })
        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["name"] == "test-policy"
        assert data["mode"] == "bidirectional"

    def test_create_policy_missing_name(self, handler: ControlPlaneHandler):
        result = handler._handle_create_federation_policy({"mode": "readonly"})
        assert result.status_code == 400

    def test_create_policy_invalid_mode(self, handler: ControlPlaneHandler):
        result = handler._handle_create_federation_policy({
            "name": "bad-policy",
            "mode": "nonexistent",
        })
        assert result.status_code == 400


class TestListFederationPolicies:
    def test_list_default_policy(self, handler: ControlPlaneHandler):
        result = handler._handle_list_federation_policies({})
        assert result.status_code == 200
        data = json.loads(result.body)
        # Should include at least the default policy
        assert data["total"] >= 1

    def test_list_after_create(self, handler: ControlPlaneHandler, coordinator: CrossWorkspaceCoordinator):
        policy = FederationPolicy(name="custom", mode=FederationMode.BIDIRECTIONAL)
        coordinator.set_policy(policy, workspace_id="ws-1")
        result = handler._handle_list_federation_policies({})
        data = json.loads(result.body)
        # default + workspace-specific
        assert data["total"] >= 2


# ============================================================================
# Execution Endpoints
# ============================================================================


class TestExecute:
    def test_execute_missing_fields(self, handler: ControlPlaneHandler):
        result = handler._handle_execute({"operation": "read_knowledge"})
        assert result.status_code == 400

    def test_execute_workspace_not_found(self, handler: ControlPlaneHandler):
        result = handler._handle_execute({
            "operation": "read_knowledge",
            "source_workspace_id": "ws-missing",
            "target_workspace_id": "ws-also-missing",
        })
        # Should return 422 because the coordinator returns failure
        data = json.loads(result.body)
        assert data["success"] is False

    def test_execute_invalid_operation(self, handler: ControlPlaneHandler):
        result = handler._handle_execute({
            "operation": "not_real_op",
            "source_workspace_id": "ws-1",
            "target_workspace_id": "ws-2",
        })
        assert result.status_code == 400


class TestListExecutions:
    def test_list_empty(self, handler: ControlPlaneHandler):
        result = handler._handle_list_executions({})
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 0

    def test_list_with_filter(self, handler: ControlPlaneHandler):
        result = handler._handle_list_executions({"workspace_id": "ws-1"})
        assert result.status_code == 200


# ============================================================================
# Consent Endpoints
# ============================================================================


class TestGrantConsent:
    def test_grant_consent_success(self, handler: ControlPlaneHandler):
        result = handler._handle_grant_consent({
            "source_workspace_id": "ws-1",
            "target_workspace_id": "ws-2",
            "scope": "metadata",
            "data_types": ["debates"],
            "operations": ["read_knowledge"],
            "granted_by": "admin",
        })
        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["source_workspace_id"] == "ws-1"
        assert data["is_valid"] is True

    def test_grant_consent_missing_workspaces(self, handler: ControlPlaneHandler):
        result = handler._handle_grant_consent({"scope": "full"})
        assert result.status_code == 400


class TestRevokeConsent:
    def test_revoke_consent_not_found(self, handler: ControlPlaneHandler):
        result = handler._handle_revoke_consent("nonexistent-id", {})
        assert result.status_code == 404

    def test_revoke_consent_success(self, handler: ControlPlaneHandler, coordinator: CrossWorkspaceCoordinator):
        consent = coordinator.grant_consent(
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            scope=SharingScope.METADATA,
            data_types={"debates"},
            operations={OperationType.READ_KNOWLEDGE},
            granted_by="admin",
        )
        result = handler._handle_revoke_consent(consent.id, {})
        assert result.status_code == 200
        assert json.loads(result.body)["revoked"] is True


class TestListConsents:
    def test_list_empty(self, handler: ControlPlaneHandler):
        result = handler._handle_list_consents({})
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 0

    def test_list_with_workspace_filter(self, handler: ControlPlaneHandler, coordinator: CrossWorkspaceCoordinator):
        coordinator.grant_consent(
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            scope=SharingScope.METADATA,
            data_types={"debates"},
            operations={OperationType.READ_KNOWLEDGE},
            granted_by="admin",
        )
        result = handler._handle_list_consents({"workspace_id": "ws-1"})
        data = json.loads(result.body)
        assert data["total"] == 1


# ============================================================================
# Approval Endpoints
# ============================================================================


class TestApproveRequest:
    def test_approve_not_found(self, handler: ControlPlaneHandler):
        result = handler._handle_approve_request("nonexistent", {})
        assert result.status_code == 404

    def test_approve_success(self, handler: ControlPlaneHandler, coordinator: CrossWorkspaceCoordinator):
        # Manually add a pending request
        req = CrossWorkspaceRequest(
            operation=OperationType.READ_KNOWLEDGE,
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
        )
        coordinator._pending_requests[req.id] = req
        result = handler._handle_approve_request(req.id, {"approved_by": "admin"})
        assert result.status_code == 200
        assert json.loads(result.body)["approved"] is True


# ============================================================================
# Stats and Health Endpoints
# ============================================================================


class TestCoordinationStats:
    def test_get_stats(self, handler: ControlPlaneHandler):
        result = handler._handle_coordination_stats({})
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "total_workspaces" in data
        assert "total_consents" in data

    def test_get_stats_no_coordinator(self):
        h = ControlPlaneHandler({"control_plane_coordinator": MagicMock()})
        result = h._handle_coordination_stats({})
        assert result.status_code == 503


class TestCoordinationHealth:
    def test_health_available(self, handler: ControlPlaneHandler):
        result = handler._handle_coordination_health({})
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "healthy"

    def test_health_unavailable(self):
        h = ControlPlaneHandler({"control_plane_coordinator": MagicMock()})
        result = h._handle_coordination_health({})
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "unavailable"


# ============================================================================
# Route Dispatch
# ============================================================================


class TestRouteDispatch:
    def test_can_handle_coordination_path(self, handler: ControlPlaneHandler):
        assert handler.can_handle("/api/v1/coordination/workspaces")
        assert handler.can_handle("/api/v1/coordination/health")

    def test_get_coordination_workspaces(self, handler: ControlPlaneHandler, mock_http_handler: MagicMock):
        mock_http_handler.path = "/api/v1/coordination/workspaces"
        result = handler.handle("/api/v1/coordination/workspaces", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_get_coordination_stats(self, handler: ControlPlaneHandler, mock_http_handler: MagicMock):
        mock_http_handler.path = "/api/v1/coordination/stats"
        result = handler.handle("/api/v1/coordination/stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_get_coordination_health(self, handler: ControlPlaneHandler, mock_http_handler: MagicMock):
        mock_http_handler.path = "/api/v1/coordination/health"
        result = handler.handle("/api/v1/coordination/health", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
