"""
Tests for ExternalIntegrationsHandler RBAC protection.

Tests cover:
- RBAC permission checks for Zapier app operations
- RBAC permission checks for Make connection operations
- RBAC permission checks for n8n credential operations
- Permission denial handling
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test Isolation Fixture
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_rbac_module_state():
    """Reset RBAC module state between tests to ensure isolation.

    This prevents test pollution when tests modify RBAC_AVAILABLE or
    other module-level state.
    """
    # Reload the module to reset all module-level state
    import aragora.server.handlers.external_integrations as ext_int_module

    importlib.reload(ext_int_module)
    yield
    # Reload again after test to clean up any modifications
    importlib.reload(ext_int_module)


# ===========================================================================
# RBAC Mock Helpers
# ===========================================================================


@dataclass
class MockPermissionDecision:
    """Mock RBAC permission decision."""

    allowed: bool = True
    reason: str = "Allowed by test"


def mock_check_permission_allowed(*args, **kwargs):
    """Mock check_permission that always allows."""
    return MockPermissionDecision(allowed=True)


def mock_check_permission_denied(*args, **kwargs):
    """Mock check_permission that always denies."""
    return MockPermissionDecision(allowed=False, reason="Permission denied by test")


@dataclass
class MockUserInfo:
    """Mock user info for RBAC testing."""

    user_id: str = "user-123"
    role: str = "viewer"
    org_id: str = "org-123"


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockZapierApp:
    """Mock Zapier app for testing."""

    id: str = "app-123"
    workspace_id: str = "ws-123"
    api_key: str = "test-api-key"
    api_secret: str = "test-api-secret"
    created_at: str = "2024-01-01T00:00:00Z"
    active: bool = True
    trigger_count: int = 0
    action_count: int = 0


@dataclass
class MockMakeConnection:
    """Mock Make connection for testing."""

    id: str = "conn-123"
    workspace_id: str = "ws-123"
    api_key: str = "test-api-key"
    created_at: str = "2024-01-01T00:00:00Z"
    active: bool = True
    total_operations: int = 0
    webhooks: list = None

    def __post_init__(self):
        if self.webhooks is None:
            self.webhooks = []


@dataclass
class MockN8nCredential:
    """Mock n8n credential for testing."""

    id: str = "cred-123"
    workspace_id: str = "ws-123"
    api_key: str = "test-api-key"
    api_url: str = "http://localhost:5678"
    created_at: str = "2024-01-01T00:00:00Z"
    active: bool = True
    operation_count: int = 0
    webhooks: list = None

    def __post_init__(self):
        if self.webhooks is None:
            self.webhooks = []


class MockZapierIntegration:
    """Mock Zapier integration."""

    TRIGGER_TYPES = ["debate_start", "debate_end", "consensus_reached"]
    ACTION_TYPES = ["create_debate", "vote"]

    def list_apps(self, workspace_id=None):
        return [MockZapierApp()]

    def create_app(self, workspace_id):
        return MockZapierApp(workspace_id=workspace_id)

    def delete_app(self, app_id):
        return app_id == "app-123"


class MockMakeIntegration:
    """Mock Make integration."""

    MODULE_TYPES = ["trigger", "action"]

    def list_connections(self, workspace_id=None):
        return [MockMakeConnection()]

    def create_connection(self, workspace_id):
        return MockMakeConnection(workspace_id=workspace_id)

    def delete_connection(self, conn_id):
        return conn_id == "conn-123"


class MockN8nIntegration:
    """Mock n8n integration."""

    def list_credentials(self, workspace_id=None):
        return [MockN8nCredential()]

    def create_credential(self, workspace_id, api_url=None):
        return MockN8nCredential(
            workspace_id=workspace_id, api_url=api_url or "http://localhost:5678"
        )

    def delete_credential(self, cred_id):
        return cred_id == "cred-123"


def make_mock_handler(method: str = "GET", body: dict = None) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = {"Content-Type": "application/json"}

    if body:
        body_bytes = json.dumps(body).encode()
        handler.rfile = BytesIO(body_bytes)
        handler.headers["Content-Length"] = str(len(body_bytes))
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


def get_status(result) -> int:
    """Extract status code from handler result."""
    if hasattr(result, "status_code"):
        return result.status_code
    return 200


def get_body(result) -> dict:
    """Extract JSON body from handler result."""
    if hasattr(result, "body"):
        return json.loads(result.body)
    return {}


@pytest.fixture
def handler_context():
    """Create handler context with mocked integrations."""
    return {
        "zapier": MockZapierIntegration(),
        "make": MockMakeIntegration(),
        "n8n": MockN8nIntegration(),
    }


@pytest.fixture
def integrations_handler(handler_context):
    """Create ExternalIntegrationsHandler instance."""
    from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

    handler = ExternalIntegrationsHandler(handler_context)
    handler._zapier = MockZapierIntegration()
    handler._make = MockMakeIntegration()
    handler._n8n = MockN8nIntegration()
    return handler


# ===========================================================================
# Test RBAC Permission Checks
# ===========================================================================


class TestExternalIntegrationsRBAC:
    """Tests for RBAC permission checks in ExternalIntegrationsHandler."""

    def test_rbac_available_check(self, integrations_handler):
        """Handler should have RBAC helper methods."""
        assert hasattr(integrations_handler, "_check_permission")
        assert hasattr(integrations_handler, "_get_auth_context")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_permission_check_without_rbac(self, integrations_handler):
        """Permission check should pass when RBAC not available."""
        handler = make_mock_handler()
        result = integrations_handler._check_permission(handler, "integrations.read")
        assert result is None  # None means allowed

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_allowed,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_permission_check_allowed(self, mock_extract, integrations_handler):
        """Permission check should pass when RBAC allows."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="admin")
        handler = make_mock_handler()

        result = integrations_handler._check_permission(handler, "integrations.read")
        assert result is None  # None means allowed

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_denied,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_permission_check_denied(self, mock_extract, integrations_handler):
        """Permission check should return error when RBAC denies."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="viewer")
        handler = make_mock_handler()

        result = integrations_handler._check_permission(handler, "integrations.create")
        assert result is not None
        assert get_status(result) == 403
        assert "Permission denied" in get_body(result).get("error", "")


# ===========================================================================
# Test Zapier RBAC Integration
# ===========================================================================


class TestZapierRBACIntegration:
    """Tests for RBAC integration in Zapier handlers."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_allowed,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_list_zapier_apps_checks_permission(self, mock_extract, integrations_handler):
        """List Zapier apps should check integrations.read permission."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="admin")
        handler = make_mock_handler()

        result = integrations_handler._handle_list_zapier_apps({}, handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "apps" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_denied,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_list_zapier_apps_denied(self, mock_extract, integrations_handler):
        """List Zapier apps should deny when permission not granted."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="viewer")
        handler = make_mock_handler()

        result = integrations_handler._handle_list_zapier_apps({}, handler)

        assert result is not None
        assert get_status(result) == 403

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_allowed,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_create_zapier_app_checks_permission(self, mock_extract, integrations_handler):
        """Create Zapier app should check integrations.create permission."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="admin")
        handler = make_mock_handler()

        result = integrations_handler._handle_create_zapier_app({"workspace_id": "ws-123"}, handler)

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "app" in body
        assert "api_key" in body["app"]  # API key is returned

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_denied,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_create_zapier_app_denied(self, mock_extract, integrations_handler):
        """Create Zapier app should deny when permission not granted."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="viewer")
        handler = make_mock_handler()

        result = integrations_handler._handle_create_zapier_app({"workspace_id": "ws-123"}, handler)

        assert result is not None
        assert get_status(result) == 403

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_allowed,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_delete_zapier_app_checks_permission(self, mock_extract, integrations_handler):
        """Delete Zapier app should check integrations.delete permission."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="admin")
        handler = make_mock_handler()

        result = integrations_handler._handle_delete_zapier_app("app-123", handler)

        assert result is not None
        assert get_status(result) == 200


# ===========================================================================
# Test Make RBAC Integration
# ===========================================================================


class TestMakeRBACIntegration:
    """Tests for RBAC integration in Make handlers."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_allowed,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_create_make_connection_checks_permission(self, mock_extract, integrations_handler):
        """Create Make connection should check integrations.create permission."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="admin")
        handler = make_mock_handler()

        result = integrations_handler._handle_create_make_connection(
            {"workspace_id": "ws-123"}, handler
        )

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "connection" in body
        assert "api_key" in body["connection"]

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_denied,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_delete_make_connection_denied(self, mock_extract, integrations_handler):
        """Delete Make connection should deny when permission not granted."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="viewer")
        handler = make_mock_handler()

        result = integrations_handler._handle_delete_make_connection("conn-123", handler)

        assert result is not None
        assert get_status(result) == 403


# ===========================================================================
# Test n8n RBAC Integration
# ===========================================================================


class TestN8nRBACIntegration:
    """Tests for RBAC integration in n8n handlers."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_allowed,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_create_n8n_credential_checks_permission(self, mock_extract, integrations_handler):
        """Create n8n credential should check integrations.create permission."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="admin")
        handler = make_mock_handler()

        result = integrations_handler._handle_create_n8n_credential(
            {"workspace_id": "ws-123"}, handler
        )

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "credential" in body
        assert "api_key" in body["credential"]

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_denied,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_list_n8n_credentials_denied(self, mock_extract, integrations_handler):
        """List n8n credentials should deny when permission not granted."""
        mock_extract.return_value = MockUserInfo(user_id="user-123", role="viewer")
        handler = make_mock_handler()

        result = integrations_handler._handle_list_n8n_credentials({}, handler)

        assert result is not None
        assert get_status(result) == 403
