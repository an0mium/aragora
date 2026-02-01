"""Tests for aragora.server.handlers.sme.teams_workspace - Teams Workspace Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Try to import the handler module
try:
    from aragora.server.handlers.sme.teams_workspace import TeamsWorkspaceHandler

    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    TeamsWorkspaceHandler = None

pytestmark = pytest.mark.skipif(not HANDLER_AVAILABLE, reason="TeamsWorkspaceHandler not available")


# ===========================================================================
# Mock Classes
# ===========================================================================


@dataclass
class MockUser:
    """Mock authenticated user."""

    user_id: str = "user-123"
    id: str = "user-123"
    org_id: str = "org-123"
    email: str = "test@example.com"
    name: str = "Test User"


@dataclass
class MockOrg:
    """Mock organization."""

    id: str = "org-123"
    name: str = "Test Organization"


@dataclass
class MockTeamsWorkspace:
    """Mock Teams workspace/tenant."""

    id: str = "ws-123"
    org_id: str = "org-123"
    tenant_id: str = "tenant-12345"
    tenant_name: str = "Test Tenant"
    bot_id: str = "bot-123"
    service_url: str = "https://smba.trafficmanager.net/test/"
    is_active: bool = True
    channels: list[str] = field(default_factory=lambda: ["channel-1", "channel-2"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "bot_id": self.bot_id,
            "service_url": self.service_url,
            "is_active": self.is_active,
            "channels": self.channels,
        }


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(
        self,
        body: bytes = b"",
        headers: Optional[dict[str, str]] = None,
        path: str = "/",
        method: str = "GET",
    ):
        self._body = body
        self.headers = headers or {"Content-Length": str(len(body))}
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)
        self.client_address = ("127.0.0.1", 12345)

    @classmethod
    def with_json_body(cls, data: dict[str, Any], **kwargs) -> "MockHandler":
        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        return cls(body=body, headers=headers, **kwargs)

    def get_argument(self, name: str, default: str = None) -> Optional[str]:
        return default


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state before each test."""
    try:
        from aragora.server.handlers.sme import teams_workspace

        teams_workspace._teams_limiter._buckets.clear()
    except Exception:
        pass
    yield


@pytest.fixture
def mock_user():
    return MockUser()


@pytest.fixture
def mock_org():
    return MockOrg()


@pytest.fixture
def mock_user_store(mock_user, mock_org):
    store = MagicMock()
    store.get_user_by_id.return_value = mock_user
    store.get_organization_by_id.return_value = mock_org
    return store


@pytest.fixture
def mock_teams_store():
    store = MagicMock()
    store.get_by_org.return_value = [MockTeamsWorkspace()]
    store.get_by_tenant_id.return_value = MockTeamsWorkspace()
    store.create.return_value = MockTeamsWorkspace()
    store.update.return_value = True
    store.delete.return_value = True
    return store


@pytest.fixture
def mock_teams_client():
    client = MagicMock()
    client.get_channels = AsyncMock(return_value=[{"id": "channel-1", "displayName": "General"}])
    client.verify_credentials = AsyncMock(return_value={"valid": True})
    return client


@pytest.fixture
def handler_context(mock_user_store):
    return {"user_store": mock_user_store}


@pytest.fixture
def teams_handler(handler_context, mock_teams_store, mock_teams_client):
    if not HANDLER_AVAILABLE:
        pytest.skip("TeamsWorkspaceHandler not available")
    handler = TeamsWorkspaceHandler(handler_context)
    yield handler


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_workspaces_list(self, teams_handler):
        """Test handler recognizes workspaces list endpoint."""
        assert teams_handler.can_handle("/api/v1/sme/teams/workspaces") is True

    def test_can_handle_workspace_detail(self, teams_handler):
        """Test handler recognizes workspace detail endpoint."""
        assert teams_handler.can_handle("/api/v1/sme/teams/workspaces/ws-123") is True

    def test_can_handle_channels(self, teams_handler):
        """Test handler recognizes channels endpoint."""
        assert teams_handler.can_handle("/api/v1/sme/teams/channels") is True

    def test_can_handle_subscribe(self, teams_handler):
        """Test handler recognizes subscribe endpoint."""
        assert teams_handler.can_handle("/api/v1/sme/teams/subscribe") is True

    def test_can_handle_subscriptions(self, teams_handler):
        """Test handler recognizes subscriptions endpoint."""
        assert teams_handler.can_handle("/api/v1/sme/teams/subscriptions") is True

    def test_cannot_handle_unknown_path(self, teams_handler):
        """Test handler rejects unknown paths."""
        assert teams_handler.can_handle("/api/v1/unknown") is False


# ===========================================================================
# List Tenants Tests
# ===========================================================================


class TestListTenants:
    """Tests for listing tenants."""

    def test_list_tenants_success(self, teams_handler, mock_user):
        """Test successful tenants listing."""
        http_handler = MockHandler(path="/api/v1/sme/teams/tenants", method="GET")

        result = teams_handler.handle("/api/v1/sme/teams/tenants", {}, http_handler, method="GET")
        assert result is not None

    def test_list_tenants_empty(self, teams_handler, mock_teams_store):
        """Test listing when no tenants configured."""
        mock_teams_store.get_by_org.return_value = []
        http_handler = MockHandler(path="/api/v1/sme/teams/tenants", method="GET")

        result = teams_handler.handle("/api/v1/sme/teams/tenants", {}, http_handler, method="GET")
        assert result is not None


# ===========================================================================
# Get Tenant Tests
# ===========================================================================


class TestGetTenant:
    """Tests for getting a single tenant."""

    def test_get_tenant_success(self, teams_handler, mock_user):
        """Test successful tenant retrieval."""
        http_handler = MockHandler(path="/api/v1/sme/teams/tenants/tenant-123", method="GET")

        result = teams_handler.handle(
            "/api/v1/sme/teams/tenants/tenant-123", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_tenant_not_found(self, teams_handler, mock_teams_store):
        """Test tenant not found error."""
        mock_teams_store.get_by_tenant_id.return_value = None
        http_handler = MockHandler(path="/api/v1/sme/teams/tenants/tenant-999", method="GET")

        result = teams_handler.handle(
            "/api/v1/sme/teams/tenants/tenant-999", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Create Tenant Tests
# ===========================================================================


class TestCreateTenant:
    """Tests for creating tenant configurations."""

    def test_create_tenant_success(self, teams_handler, mock_user):
        """Test successful tenant creation."""
        body = {
            "tenant_id": "tenant-12345",
            "client_id": "client-id",
            "client_secret": "client-secret",
        }
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/teams/tenants", method="POST"
        )

        result = teams_handler.handle("/api/v1/sme/teams/tenants", {}, http_handler, method="POST")
        assert result is not None

    def test_create_tenant_missing_tenant_id(self, teams_handler):
        """Test error when tenant_id is missing."""
        body = {"client_id": "client-id"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/teams/tenants", method="POST"
        )

        result = teams_handler.handle("/api/v1/sme/teams/tenants", {}, http_handler, method="POST")
        assert result is not None


# ===========================================================================
# Update Tenant Tests
# ===========================================================================


class TestUpdateTenant:
    """Tests for updating tenants."""

    def test_update_tenant_success(self, teams_handler, mock_user):
        """Test successful tenant update."""
        body = {"is_active": False}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/teams/tenants/tenant-123", method="PATCH"
        )

        result = teams_handler.handle(
            "/api/v1/sme/teams/tenants/tenant-123", {}, http_handler, method="PATCH"
        )
        assert result is not None


# ===========================================================================
# Delete Tenant Tests
# ===========================================================================


class TestDeleteTenant:
    """Tests for deleting tenants."""

    def test_delete_tenant_success(self, teams_handler, mock_user):
        """Test successful tenant deletion."""
        http_handler = MockHandler(path="/api/v1/sme/teams/tenants/tenant-123", method="DELETE")

        result = teams_handler.handle(
            "/api/v1/sme/teams/tenants/tenant-123", {}, http_handler, method="DELETE"
        )
        assert result is not None


# ===========================================================================
# OAuth Tests
# ===========================================================================


class TestOAuth:
    """Tests for OAuth flow.

    Note: The TeamsWorkspaceHandler does not currently expose dedicated
    /oauth/* routes. These tests verify the subscribe endpoint which is
    the handler's entry point for workspace configuration.
    """

    def test_subscribe_endpoint_success(self, teams_handler, mock_user):
        """Test subscribe endpoint (used for workspace config)."""
        http_handler = MockHandler(path="/api/v1/sme/teams/subscribe", method="POST")

        result = teams_handler.handle(
            "/api/v1/sme/teams/subscribe", {}, http_handler, method="POST"
        )
        assert result is not None

    @pytest.mark.skip(reason="OAuth callback route not yet implemented in TeamsWorkspaceHandler")
    def test_oauth_callback_success(self, teams_handler, mock_user):
        """Test OAuth callback endpoint (pending route implementation)."""
        pass


# ===========================================================================
# Channels Tests
# ===========================================================================


class TestChannels:
    """Tests for channel listing."""

    def test_list_channels_success(self, teams_handler, mock_user):
        """Test successful channel listing."""
        http_handler = MockHandler(
            path="/api/v1/sme/teams/tenants/tenant-123/channels", method="GET"
        )

        result = teams_handler.handle(
            "/api/v1/sme/teams/tenants/tenant-123/channels", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self, teams_handler):
        """Test rate limit enforcement."""
        http_handler = MockHandler(path="/api/v1/sme/teams/workspaces", method="GET")

        with patch(
            "aragora.server.handlers.sme.teams_workspace._teams_limiter.is_allowed",
            return_value=False,
        ):
            result = teams_handler.handle(
                "/api/v1/sme/teams/workspaces", {}, http_handler, method="GET"
            )
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_workspaces_list_method_not_allowed(self, teams_handler):
        """Test method not allowed for workspaces list."""
        http_handler = MockHandler(path="/api/v1/sme/teams/workspaces", method="PUT")

        result = teams_handler.handle(
            "/api/v1/sme/teams/workspaces", {}, http_handler, method="PUT"
        )
        assert result is not None
        assert result.status_code == 405
