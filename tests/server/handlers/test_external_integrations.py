"""
Comprehensive tests for ExternalIntegrationsHandler.

Tests cover:
- Integration CRUD operations (Zapier, Make, n8n)
- Connection testing
- Webhook configuration and sync operations
- Trigger subscriptions
- Error handling paths
- Input validation
- Rate limiting behavior
- Authorization checks
- Platform-specific operations
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ===========================================================================
# Test Isolation Fixture
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module state between tests to ensure isolation."""
    import aragora.server.handlers.external_integrations as ext_int_module

    importlib.reload(ext_int_module)
    yield
    importlib.reload(ext_int_module)


# ===========================================================================
# Mock Data Classes
# ===========================================================================


@dataclass
class MockPermissionDecision:
    """Mock RBAC permission decision."""

    allowed: bool = True
    reason: str = "Allowed by test"


@dataclass
class MockUserInfo:
    """Mock user info for RBAC testing."""

    user_id: str = "user-123"
    role: str = "admin"
    org_id: str = "org-123"


@dataclass
class MockZapierApp:
    """Mock Zapier app for testing."""

    id: str = "app-123"
    workspace_id: str = "ws-123"
    api_key: str = "test-api-key"
    api_secret: str = "test-api-secret"
    created_at: str = "2024-01-01T00:00:00Z"
    active: bool = True
    trigger_count: int = 5
    action_count: int = 10


@dataclass
class MockZapierTrigger:
    """Mock Zapier trigger for testing."""

    id: str = "trigger-123"
    trigger_type: str = "debate_completed"
    webhook_url: str = "https://hooks.zapier.com/test"
    created_at: str = "2024-01-01T00:00:00Z"


@dataclass
class MockMakeConnection:
    """Mock Make connection for testing."""

    id: str = "conn-123"
    workspace_id: str = "ws-123"
    api_key: str = "test-api-key"
    created_at: str = "2024-01-01T00:00:00Z"
    active: bool = True
    total_operations: int = 25
    webhooks: list = field(default_factory=list)


@dataclass
class MockMakeWebhook:
    """Mock Make webhook for testing."""

    id: str = "webhook-123"
    module_type: str = "watch_debates"
    webhook_url: str = "https://hook.make.com/test"
    created_at: str = "2024-01-01T00:00:00Z"


@dataclass
class MockN8nCredential:
    """Mock n8n credential for testing."""

    id: str = "cred-123"
    workspace_id: str = "ws-123"
    api_key: str = "test-api-key"
    api_url: str = "http://localhost:5678"
    created_at: str = "2024-01-01T00:00:00Z"
    active: bool = True
    operation_count: int = 15
    webhooks: list = field(default_factory=list)


@dataclass
class MockN8nWebhook:
    """Mock n8n webhook for testing."""

    id: str = "webhook-456"
    webhook_path: str = "/n8n/webhook/webhook-456"
    events: list = field(default_factory=lambda: ["debate_end", "consensus"])
    created_at: str = "2024-01-01T00:00:00Z"


# ===========================================================================
# Mock Integration Classes
# ===========================================================================


class MockZapierIntegration:
    """Mock Zapier integration with configurable behavior."""

    TRIGGER_TYPES = {
        "debate_completed": "Fires when a debate finishes",
        "consensus_reached": "Fires when agents reach consensus",
        "gauntlet_completed": "Fires when stress-test completes",
    }
    ACTION_TYPES = {
        "start_debate": "Start a new multi-agent debate",
        "get_debate": "Get debate status and results",
    }

    def __init__(self):
        self._apps: dict[str, MockZapierApp] = {
            "app-123": MockZapierApp(),
        }
        self._triggers: dict[str, MockZapierTrigger] = {}

    def list_apps(self, workspace_id=None):
        apps = list(self._apps.values())
        if workspace_id:
            apps = [a for a in apps if a.workspace_id == workspace_id]
        return apps

    def create_app(self, workspace_id):
        return MockZapierApp(workspace_id=workspace_id, id=f"app-{workspace_id}")

    def delete_app(self, app_id):
        return app_id in self._apps

    def subscribe_trigger(
        self,
        app_id,
        trigger_type,
        webhook_url,
        workspace_id=None,
        debate_tags=None,
        min_confidence=None,
    ):
        if app_id not in self._apps:
            return None
        if trigger_type not in self.TRIGGER_TYPES:
            return None
        return MockZapierTrigger(trigger_type=trigger_type, webhook_url=webhook_url)

    def unsubscribe_trigger(self, app_id, trigger_id):
        return app_id in self._apps and trigger_id == "trigger-123"


class MockMakeIntegration:
    """Mock Make integration with configurable behavior."""

    MODULE_TYPES = {
        "watch_debates": {"type": "trigger", "instant": True},
        "watch_consensus": {"type": "trigger", "instant": True},
        "create_debate": {"type": "action"},
        "get_debate": {"type": "action"},
    }

    def __init__(self):
        self._connections: dict[str, MockMakeConnection] = {
            "conn-123": MockMakeConnection(),
        }

    def list_connections(self, workspace_id=None):
        conns = list(self._connections.values())
        if workspace_id:
            conns = [c for c in conns if c.workspace_id == workspace_id]
        return conns

    def create_connection(self, workspace_id):
        return MockMakeConnection(workspace_id=workspace_id, id=f"conn-{workspace_id}")

    def delete_connection(self, conn_id):
        return conn_id in self._connections

    def register_webhook(
        self,
        conn_id,
        module_type,
        webhook_url,
        workspace_id=None,
        event_filter=None,
    ):
        if conn_id not in self._connections:
            return None
        if module_type not in self.MODULE_TYPES:
            return None
        return MockMakeWebhook(module_type=module_type, webhook_url=webhook_url)

    def unregister_webhook(self, conn_id, webhook_id):
        return conn_id in self._connections and webhook_id == "webhook-123"


class MockN8nIntegration:
    """Mock n8n integration with configurable behavior."""

    EVENT_TYPES = {
        "debate_start": "Debate has started",
        "debate_end": "Debate has completed",
        "consensus": "Consensus reached",
        "decision_made": "Final decision recorded",
    }

    def __init__(self):
        self._credentials: dict[str, MockN8nCredential] = {
            "cred-123": MockN8nCredential(),
        }

    def list_credentials(self, workspace_id=None):
        creds = list(self._credentials.values())
        if workspace_id:
            creds = [c for c in creds if c.workspace_id == workspace_id]
        return creds

    def create_credential(self, workspace_id, api_url=None):
        return MockN8nCredential(
            workspace_id=workspace_id,
            api_url=api_url or "http://localhost:5678",
            id=f"cred-{workspace_id}",
        )

    def delete_credential(self, cred_id):
        return cred_id in self._credentials

    def get_node_definition(self):
        return {
            "displayName": "Aragora",
            "name": "aragora",
            "version": 1,
        }

    def get_trigger_node_definition(self):
        return {
            "displayName": "Aragora Trigger",
            "name": "aragoraTrigger",
            "version": 1,
        }

    def get_credential_definition(self):
        return {
            "name": "aragoraApi",
            "displayName": "Aragora API",
        }

    def register_webhook(
        self,
        cred_id,
        events,
        workflow_id=None,
        node_id=None,
        workspace_id=None,
    ):
        if cred_id not in self._credentials:
            return None
        return MockN8nWebhook(events=events)

    def unregister_webhook(self, cred_id, webhook_id):
        return cred_id in self._credentials and webhook_id == "webhook-456"


# ===========================================================================
# Helper Functions
# ===========================================================================


def make_mock_handler(method: str = "GET", body: dict = None, headers: dict = None) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = headers or {"Content-Type": "application/json"}

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
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode())
        return json.loads(result.body)
    return {}


def mock_check_permission_allowed(*args, **kwargs):
    """Mock check_permission that always allows."""
    return MockPermissionDecision(allowed=True)


def mock_check_permission_denied(*args, **kwargs):
    """Mock check_permission that always denies."""
    return MockPermissionDecision(allowed=False, reason="Permission denied")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def server_context():
    """Create basic server context."""
    return {}


@pytest.fixture
def mock_integrations():
    """Create mock integration instances."""
    return {
        "zapier": MockZapierIntegration(),
        "make": MockMakeIntegration(),
        "n8n": MockN8nIntegration(),
    }


@pytest.fixture
def integrations_handler(server_context, mock_integrations):
    """Create ExternalIntegrationsHandler with mocked integrations."""
    from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

    handler = ExternalIntegrationsHandler(server_context)
    handler._zapier = mock_integrations["zapier"]
    handler._make = mock_integrations["make"]
    handler._n8n = mock_integrations["n8n"]
    return handler


# ===========================================================================
# Test: can_handle Routing
# ===========================================================================


class TestCanHandleRouting:
    """Tests for the can_handle static method."""

    def test_handles_zapier_apps_path(self):
        """Should handle Zapier apps endpoint."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert ExternalIntegrationsHandler.can_handle("/api/integrations/zapier/apps")

    def test_handles_zapier_triggers_path(self):
        """Should handle Zapier triggers endpoint."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert ExternalIntegrationsHandler.can_handle("/api/integrations/zapier/triggers")

    def test_handles_make_connections_path(self):
        """Should handle Make connections endpoint."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert ExternalIntegrationsHandler.can_handle("/api/integrations/make/connections")

    def test_handles_make_webhooks_path(self):
        """Should handle Make webhooks endpoint."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert ExternalIntegrationsHandler.can_handle("/api/integrations/make/webhooks")

    def test_handles_n8n_credentials_path(self):
        """Should handle n8n credentials endpoint."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert ExternalIntegrationsHandler.can_handle("/api/integrations/n8n/credentials")

    def test_handles_n8n_nodes_path(self):
        """Should handle n8n nodes endpoint."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert ExternalIntegrationsHandler.can_handle("/api/integrations/n8n/nodes")

    def test_handles_versioned_paths(self):
        """Should handle versioned API paths."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert ExternalIntegrationsHandler.can_handle("/api/v1/integrations/zapier/apps")
        assert ExternalIntegrationsHandler.can_handle("/api/v2/integrations/make/connections")

    def test_rejects_non_integration_paths(self):
        """Should reject non-integration paths."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert not ExternalIntegrationsHandler.can_handle("/api/debates")
        assert not ExternalIntegrationsHandler.can_handle("/api/users")
        assert not ExternalIntegrationsHandler.can_handle("/api/health")

    def test_rejects_unknown_platforms(self):
        """Should reject unknown integration platforms."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert not ExternalIntegrationsHandler.can_handle("/api/integrations/unknown/apps")
        assert not ExternalIntegrationsHandler.can_handle("/api/integrations/slack/channels")


# ===========================================================================
# Test: Zapier CRUD Operations
# ===========================================================================


class TestZapierCRUDOperations:
    """Tests for Zapier app CRUD operations."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_list_zapier_apps_returns_apps(self, integrations_handler):
        """List Zapier apps should return app list."""
        handler = make_mock_handler()
        result = integrations_handler._handle_list_zapier_apps({}, handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "apps" in body
        assert "count" in body
        assert body["count"] >= 1

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_list_zapier_apps_filters_by_workspace(self, integrations_handler):
        """List Zapier apps should filter by workspace_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_list_zapier_apps(
            {"workspace_id": ["ws-123"]}, handler
        )

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "apps" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_zapier_app_success(self, integrations_handler):
        """Create Zapier app should succeed with workspace_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_create_zapier_app({"workspace_id": "ws-new"}, handler)

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "app" in body
        assert "api_key" in body["app"]
        assert "api_secret" in body["app"]
        assert "message" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_zapier_app_missing_workspace(self, integrations_handler):
        """Create Zapier app should fail without workspace_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_create_zapier_app({}, handler)

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "error" in body
        assert "workspace_id" in body["error"]

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_delete_zapier_app_success(self, integrations_handler):
        """Delete Zapier app should succeed for existing app."""
        handler = make_mock_handler()
        result = integrations_handler._handle_delete_zapier_app("app-123", handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("deleted") is True

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_delete_zapier_app_not_found(self, integrations_handler):
        """Delete Zapier app should return 404 for non-existent app."""
        handler = make_mock_handler()
        result = integrations_handler._handle_delete_zapier_app("nonexistent", handler)

        assert result is not None
        assert get_status(result) == 404


# ===========================================================================
# Test: Zapier Trigger Operations
# ===========================================================================


class TestZapierTriggerOperations:
    """Tests for Zapier trigger subscription operations."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_list_trigger_types(self, integrations_handler):
        """Should return available trigger types."""
        handler = make_mock_handler()
        result = integrations_handler._handle_list_zapier_trigger_types(handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "triggers" in body
        assert "actions" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_subscribe_trigger_success(self, integrations_handler):
        """Subscribe trigger should succeed with valid parameters."""
        handler = make_mock_handler()
        result = integrations_handler._handle_subscribe_zapier_trigger(
            {
                "app_id": "app-123",
                "trigger_type": "debate_completed",
                "webhook_url": "https://hooks.zapier.com/test",
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "trigger" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_subscribe_trigger_missing_app_id(self, integrations_handler):
        """Subscribe trigger should fail without app_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_subscribe_zapier_trigger(
            {
                "trigger_type": "debate_completed",
                "webhook_url": "https://hooks.zapier.com/test",
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "app_id" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_subscribe_trigger_missing_trigger_type(self, integrations_handler):
        """Subscribe trigger should fail without trigger_type."""
        handler = make_mock_handler()
        result = integrations_handler._handle_subscribe_zapier_trigger(
            {
                "app_id": "app-123",
                "webhook_url": "https://hooks.zapier.com/test",
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "trigger_type" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_subscribe_trigger_missing_webhook_url(self, integrations_handler):
        """Subscribe trigger should fail without webhook_url."""
        handler = make_mock_handler()
        result = integrations_handler._handle_subscribe_zapier_trigger(
            {
                "app_id": "app-123",
                "trigger_type": "debate_completed",
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "webhook_url" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_unsubscribe_trigger_success(self, integrations_handler):
        """Unsubscribe trigger should succeed for existing trigger."""
        handler = make_mock_handler()
        result = integrations_handler._handle_unsubscribe_zapier_trigger(
            "app-123", "trigger-123", handler
        )

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("deleted") is True

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_unsubscribe_trigger_missing_app_id(self, integrations_handler):
        """Unsubscribe trigger should fail without app_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_unsubscribe_zapier_trigger("", "trigger-123", handler)

        assert result is not None
        assert get_status(result) == 400


# ===========================================================================
# Test: Make CRUD Operations
# ===========================================================================


class TestMakeCRUDOperations:
    """Tests for Make connection CRUD operations."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_list_make_connections(self, integrations_handler):
        """List Make connections should return connection list."""
        handler = make_mock_handler()
        result = integrations_handler._handle_list_make_connections({}, handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "connections" in body
        assert "count" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_list_make_modules(self, integrations_handler):
        """List Make modules should return module types."""
        handler = make_mock_handler()
        result = integrations_handler._handle_list_make_modules(handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "modules" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_make_connection_success(self, integrations_handler):
        """Create Make connection should succeed with workspace_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_create_make_connection(
            {"workspace_id": "ws-new"}, handler
        )

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "connection" in body
        assert "api_key" in body["connection"]

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_make_connection_missing_workspace(self, integrations_handler):
        """Create Make connection should fail without workspace_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_create_make_connection({}, handler)

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "workspace_id" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_delete_make_connection_success(self, integrations_handler):
        """Delete Make connection should succeed for existing connection."""
        handler = make_mock_handler()
        result = integrations_handler._handle_delete_make_connection("conn-123", handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("deleted") is True

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_delete_make_connection_not_found(self, integrations_handler):
        """Delete Make connection should return 404 for non-existent."""
        handler = make_mock_handler()
        result = integrations_handler._handle_delete_make_connection("nonexistent", handler)

        assert result is not None
        assert get_status(result) == 404


# ===========================================================================
# Test: Make Webhook Operations
# ===========================================================================


class TestMakeWebhookOperations:
    """Tests for Make webhook registration operations."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_register_webhook_success(self, integrations_handler):
        """Register Make webhook should succeed with valid parameters."""
        handler = make_mock_handler()
        result = integrations_handler._handle_register_make_webhook(
            {
                "connection_id": "conn-123",
                "module_type": "watch_debates",
                "webhook_url": "https://hook.make.com/test",
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "webhook" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_register_webhook_missing_connection_id(self, integrations_handler):
        """Register webhook should fail without connection_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_register_make_webhook(
            {
                "module_type": "watch_debates",
                "webhook_url": "https://hook.make.com/test",
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "connection_id" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_register_webhook_missing_module_type(self, integrations_handler):
        """Register webhook should fail without module_type."""
        handler = make_mock_handler()
        result = integrations_handler._handle_register_make_webhook(
            {
                "connection_id": "conn-123",
                "webhook_url": "https://hook.make.com/test",
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "module_type" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_unregister_webhook_success(self, integrations_handler):
        """Unregister Make webhook should succeed for existing webhook."""
        handler = make_mock_handler()
        result = integrations_handler._handle_unregister_make_webhook(
            "conn-123", "webhook-123", handler
        )

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("deleted") is True

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_unregister_webhook_missing_connection_id(self, integrations_handler):
        """Unregister webhook should fail without connection_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_unregister_make_webhook("", "webhook-123", handler)

        assert result is not None
        assert get_status(result) == 400


# ===========================================================================
# Test: n8n CRUD Operations
# ===========================================================================


class TestN8nCRUDOperations:
    """Tests for n8n credential CRUD operations."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_list_n8n_credentials(self, integrations_handler):
        """List n8n credentials should return credential list."""
        handler = make_mock_handler()
        result = integrations_handler._handle_list_n8n_credentials({}, handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "credentials" in body
        assert "count" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_get_n8n_nodes(self, integrations_handler):
        """Get n8n nodes should return node definitions."""
        handler = make_mock_handler()
        result = integrations_handler._handle_get_n8n_nodes(handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "node" in body
        assert "trigger" in body
        assert "credential" in body
        assert "events" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_n8n_credential_success(self, integrations_handler):
        """Create n8n credential should succeed with workspace_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_create_n8n_credential(
            {"workspace_id": "ws-new"}, handler
        )

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "credential" in body
        assert "api_key" in body["credential"]
        assert "api_url" in body["credential"]

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_n8n_credential_with_api_url(self, integrations_handler):
        """Create n8n credential should accept custom api_url."""
        handler = make_mock_handler()
        result = integrations_handler._handle_create_n8n_credential(
            {"workspace_id": "ws-new", "api_url": "https://custom.n8n.io"}, handler
        )

        assert result is not None
        assert get_status(result) == 201

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_n8n_credential_missing_workspace(self, integrations_handler):
        """Create n8n credential should fail without workspace_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_create_n8n_credential({}, handler)

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "workspace_id" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_delete_n8n_credential_success(self, integrations_handler):
        """Delete n8n credential should succeed for existing credential."""
        handler = make_mock_handler()
        result = integrations_handler._handle_delete_n8n_credential("cred-123", handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("deleted") is True

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_delete_n8n_credential_not_found(self, integrations_handler):
        """Delete n8n credential should return 404 for non-existent."""
        handler = make_mock_handler()
        result = integrations_handler._handle_delete_n8n_credential("nonexistent", handler)

        assert result is not None
        assert get_status(result) == 404


# ===========================================================================
# Test: n8n Webhook Operations
# ===========================================================================


class TestN8nWebhookOperations:
    """Tests for n8n webhook registration operations."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_register_n8n_webhook_success(self, integrations_handler):
        """Register n8n webhook should succeed with valid parameters."""
        handler = make_mock_handler()
        result = integrations_handler._handle_register_n8n_webhook(
            {
                "credential_id": "cred-123",
                "events": ["debate_end", "consensus"],
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 201
        body = get_body(result)
        assert "webhook" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_register_n8n_webhook_missing_credential_id(self, integrations_handler):
        """Register n8n webhook should fail without credential_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_register_n8n_webhook(
            {"events": ["debate_end"]}, handler
        )

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "credential_id" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_register_n8n_webhook_missing_events(self, integrations_handler):
        """Register n8n webhook should fail without events."""
        handler = make_mock_handler()
        result = integrations_handler._handle_register_n8n_webhook(
            {"credential_id": "cred-123"}, handler
        )

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "events" in body.get("error", "")

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_unregister_n8n_webhook_success(self, integrations_handler):
        """Unregister n8n webhook should succeed for existing webhook."""
        handler = make_mock_handler()
        result = integrations_handler._handle_unregister_n8n_webhook(
            "cred-123", "webhook-456", handler
        )

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("deleted") is True

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_unregister_n8n_webhook_missing_credential_id(self, integrations_handler):
        """Unregister n8n webhook should fail without credential_id."""
        handler = make_mock_handler()
        result = integrations_handler._handle_unregister_n8n_webhook("", "webhook-456", handler)

        assert result is not None
        assert get_status(result) == 400


# ===========================================================================
# Test: Integration Testing Endpoints
# ===========================================================================


class TestIntegrationTestEndpoints:
    """Tests for platform connection testing."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_test_zapier_integration(self, integrations_handler):
        """Test Zapier integration should return status."""
        handler = make_mock_handler()
        result = integrations_handler._handle_test_integration("zapier", handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("platform") == "zapier"
        assert body.get("status") == "ok"
        assert "apps_count" in body
        assert "trigger_types" in body
        assert "action_types" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_test_make_integration(self, integrations_handler):
        """Test Make integration should return status."""
        handler = make_mock_handler()
        result = integrations_handler._handle_test_integration("make", handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("platform") == "make"
        assert body.get("status") == "ok"
        assert "connections_count" in body
        assert "module_types" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_test_n8n_integration(self, integrations_handler):
        """Test n8n integration should return status."""
        handler = make_mock_handler()
        result = integrations_handler._handle_test_integration("n8n", handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body.get("platform") == "n8n"
        assert body.get("status") == "ok"
        assert "credentials_count" in body
        assert "event_types" in body

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_test_unknown_platform(self, integrations_handler):
        """Test unknown platform should return 400."""
        handler = make_mock_handler()
        result = integrations_handler._handle_test_integration("unknown", handler)

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "Unknown platform" in body.get("error", "")


# ===========================================================================
# Test: RBAC Permission Checks
# ===========================================================================


class TestRBACPermissions:
    """Tests for RBAC permission enforcement."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_denied,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_permission_denied_returns_403(self, mock_extract, integrations_handler):
        """Permission denial should return 403."""
        mock_extract.return_value = MockUserInfo(role="viewer")
        handler = make_mock_handler()

        result = integrations_handler._check_permission(handler, "integrations.write")

        assert result is not None
        assert get_status(result) == 403

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.external_integrations.check_permission",
        mock_check_permission_allowed,
    )
    @patch("aragora.server.handlers.external_integrations.extract_user_from_request")
    def test_permission_allowed_returns_none(self, mock_extract, integrations_handler):
        """Permission allowed should return None (pass-through)."""
        mock_extract.return_value = MockUserInfo(role="admin")
        handler = make_mock_handler()

        result = integrations_handler._check_permission(handler, "integrations.read")

        assert result is None

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_rbac_unavailable_allows_all(self, integrations_handler):
        """When RBAC unavailable, all operations should be allowed."""
        handler = make_mock_handler()

        result = integrations_handler._check_permission(handler, "integrations.write")

        assert result is None


# ===========================================================================
# Test: Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limiter_constants_defined(self):
        """Rate limiter constants should be defined."""
        from aragora.server.handlers.external_integrations import (
            INTEGRATION_CREATE_RPM,
            INTEGRATION_LIST_RPM,
            INTEGRATION_DELETE_RPM,
            INTEGRATION_TEST_RPM,
        )

        assert INTEGRATION_CREATE_RPM > 0
        assert INTEGRATION_LIST_RPM > 0
        assert INTEGRATION_DELETE_RPM > 0
        assert INTEGRATION_TEST_RPM > 0

    def test_rate_limiters_initialized(self):
        """Rate limiters should be initialized."""
        from aragora.server.handlers.external_integrations import (
            _create_limiter,
            _list_limiter,
            _delete_limiter,
            _test_limiter,
        )

        assert _create_limiter is not None
        assert _list_limiter is not None
        assert _delete_limiter is not None
        assert _test_limiter is not None


# ===========================================================================
# Test: Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling paths."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_with_invalid_json_body(self, integrations_handler):
        """Handler should handle missing body gracefully."""
        # This tests the validation in handle_post which calls read_json_body_validated
        handler = make_mock_handler(body=None)

        # The handler would need the body param to be passed in
        # This simulates the flow when no body is provided
        result = integrations_handler._handle_create_zapier_app({}, handler)

        assert result is not None
        # Without workspace_id, should return 400
        assert get_status(result) == 400

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_subscribe_trigger_invalid_app_returns_failure(self, integrations_handler):
        """Subscribe to trigger for invalid app should fail gracefully."""
        handler = make_mock_handler()
        result = integrations_handler._handle_subscribe_zapier_trigger(
            {
                "app_id": "nonexistent-app",
                "trigger_type": "debate_completed",
                "webhook_url": "https://hooks.zapier.com/test",
            },
            handler,
        )

        assert result is not None
        # The mock returns None for invalid app, handler should return 400
        assert get_status(result) == 400

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_register_webhook_invalid_module_type(self, integrations_handler):
        """Register webhook with invalid module type should fail."""
        handler = make_mock_handler()
        result = integrations_handler._handle_register_make_webhook(
            {
                "connection_id": "conn-123",
                "module_type": "invalid_module",
                "webhook_url": "https://hook.make.com/test",
            },
            handler,
        )

        assert result is not None
        # The mock returns None for invalid module type
        assert get_status(result) == 400

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_unsubscribe_nonexistent_trigger(self, integrations_handler):
        """Unsubscribe from non-existent trigger should return 404."""
        handler = make_mock_handler()
        result = integrations_handler._handle_unsubscribe_zapier_trigger(
            "app-123", "nonexistent-trigger", handler
        )

        assert result is not None
        assert get_status(result) == 404


# ===========================================================================
# Test: Handler Routes Definition
# ===========================================================================


class TestHandlerRoutes:
    """Tests for handler route definitions."""

    def test_routes_defined(self):
        """Handler should have routes defined."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        assert hasattr(ExternalIntegrationsHandler, "routes")
        assert len(ExternalIntegrationsHandler.routes) > 0

    def test_routes_include_zapier(self):
        """Routes should include Zapier endpoints."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        routes_str = " ".join(ExternalIntegrationsHandler.routes)
        assert "zapier" in routes_str

    def test_routes_include_make(self):
        """Routes should include Make endpoints."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        routes_str = " ".join(ExternalIntegrationsHandler.routes)
        assert "make" in routes_str

    def test_routes_include_n8n(self):
        """Routes should include n8n endpoints."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        routes_str = " ".join(ExternalIntegrationsHandler.routes)
        assert "n8n" in routes_str

    def test_routes_include_test_endpoint(self):
        """Routes should include test endpoint."""
        from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler

        routes_str = " ".join(ExternalIntegrationsHandler.routes)
        assert "/test" in routes_str


# ===========================================================================
# Test: Integration Instance Management
# ===========================================================================


class TestIntegrationInstanceManagement:
    """Tests for lazy integration instance management."""

    def test_get_zapier_returns_instance(self, integrations_handler):
        """_get_zapier should return integration instance."""
        result = integrations_handler._get_zapier()
        assert result is not None

    def test_get_make_returns_instance(self, integrations_handler):
        """_get_make should return integration instance."""
        result = integrations_handler._get_make()
        assert result is not None

    def test_get_n8n_returns_instance(self, integrations_handler):
        """_get_n8n should return integration instance."""
        result = integrations_handler._get_n8n()
        assert result is not None


# ===========================================================================
# Test: Response Format Validation
# ===========================================================================


class TestResponseFormat:
    """Tests for response format consistency."""

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_list_response_has_count_field(self, integrations_handler):
        """List responses should include count field."""
        handler = make_mock_handler()

        # Test all list endpoints
        zapier_result = integrations_handler._handle_list_zapier_apps({}, handler)
        make_result = integrations_handler._handle_list_make_connections({}, handler)
        n8n_result = integrations_handler._handle_list_n8n_credentials({}, handler)

        assert "count" in get_body(zapier_result)
        assert "count" in get_body(make_result)
        assert "count" in get_body(n8n_result)

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_create_response_includes_message(self, integrations_handler):
        """Create responses should include user-friendly message."""
        handler = make_mock_handler()

        zapier_result = integrations_handler._handle_create_zapier_app(
            {"workspace_id": "ws-1"}, handler
        )
        make_result = integrations_handler._handle_create_make_connection(
            {"workspace_id": "ws-1"}, handler
        )
        n8n_result = integrations_handler._handle_create_n8n_credential(
            {"workspace_id": "ws-1"}, handler
        )

        assert "message" in get_body(zapier_result)
        assert "message" in get_body(make_result)
        assert "message" in get_body(n8n_result)

    @patch("aragora.server.handlers.external_integrations.RBAC_AVAILABLE", False)
    def test_delete_response_includes_deleted_flag(self, integrations_handler):
        """Delete responses should include deleted flag."""
        handler = make_mock_handler()

        zapier_result = integrations_handler._handle_delete_zapier_app("app-123", handler)
        make_result = integrations_handler._handle_delete_make_connection("conn-123", handler)
        n8n_result = integrations_handler._handle_delete_n8n_credential("cred-123", handler)

        assert get_body(zapier_result).get("deleted") is True
        assert get_body(make_result).get("deleted") is True
        assert get_body(n8n_result).get("deleted") is True
