"""Tests for Teams Workspace Handler."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.sme.teams_workspace import TeamsWorkspaceHandler


@dataclass
class MockUser:
    """Mock user for testing."""

    user_id: str = "user-123"
    id: str = "user-123"
    org_id: str = "org-456"
    email: str = "test@example.com"


@dataclass
class MockOrg:
    """Mock organization for testing."""

    id: str = "org-456"
    name: str = "Test Org"
    slug: str = "test-org"


@dataclass
class MockTeamsWorkspace:
    """Mock Teams workspace for testing."""

    tenant_id: str = "tenant-123"
    tenant_name: str = "Test Tenant"
    access_token: str = "test-token-12345"
    bot_id: str = "bot-123"
    installed_at: float = 1700000000.0
    installed_by: Optional[str] = "user-123"
    scopes: list[str] = None
    aragora_tenant_id: str = "org-456"
    is_active: bool = True
    refresh_token: Optional[str] = None
    token_expires_at: Optional[float] = None
    service_url: Optional[str] = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = ["Team.ReadBasic.All"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "bot_id": self.bot_id,
            "installed_at": self.installed_at,
            "installed_at_iso": datetime.fromtimestamp(
                self.installed_at, tz=timezone.utc
            ).isoformat(),
            "installed_by": self.installed_by,
            "scopes": self.scopes,
            "aragora_tenant_id": self.aragora_tenant_id,
            "is_active": self.is_active,
            "has_refresh_token": bool(self.refresh_token),
            "token_expires_at": self.token_expires_at,
            "service_url": self.service_url,
        }


class MockRequest(dict):
    """Mock HTTP request handler that also acts as a dict for query params.

    Note: get_string_param(handler, key, default) expects handler to have a .get() method.
    By inheriting from dict, this mock can store query params and be passed directly
    to get_string_param().
    """

    def __init__(
        self,
        command: str = "GET",
        path: str = "/",
        headers: Optional[dict[str, str]] = None,
        body: Optional[bytes] = None,
        query_params: Optional[dict[str, str]] = None,
    ):
        # Initialize dict with query params
        super().__init__(query_params or {})
        self.command = command
        self.path = path
        self.headers = headers or {"Content-Length": "0"}
        self._body = body or b""
        self.rfile = BytesIO(self._body)
        if body:
            self.headers["Content-Length"] = str(len(body))
        self.client_address = ("127.0.0.1", 12345)


@pytest.fixture
def mock_ctx():
    """Create mock server context."""
    user_store = MagicMock()
    user_store.get_user_by_id.return_value = MockUser()
    user_store.get_organization_by_id.return_value = MockOrg()

    return {"user_store": user_store}


@pytest.fixture
def mock_workspace_store():
    """Create mock workspace store."""
    store = MagicMock()
    store.get.return_value = None
    store.get_by_aragora_tenant.return_value = []
    store.deactivate.return_value = True
    return store


@pytest.fixture
def mock_subscription_store():
    """Create mock subscription store."""
    store = MagicMock()
    store.get.return_value = None
    store.get_by_org.return_value = []
    store.create.return_value = MagicMock(
        id="sub-123",
        to_dict=lambda: {
            "id": "sub-123",
            "org_id": "org-456",
            "channel_type": "teams",
            "channel_id": "channel-123",
        },
    )
    store.delete.return_value = True
    store.deactivate.return_value = True
    return store


@pytest.fixture
def handler(mock_ctx, mock_workspace_store, mock_subscription_store):
    """Create handler with mocked dependencies."""
    h = TeamsWorkspaceHandler(mock_ctx)

    # Patch store methods
    h._get_workspace_store = MagicMock(return_value=mock_workspace_store)
    h._get_subscription_store = MagicMock(return_value=mock_subscription_store)

    return h


class TestTeamsWorkspaceHandler:
    """Tests for TeamsWorkspaceHandler."""

    def test_can_handle_static_routes(self, handler):
        """Test can_handle for static routes."""
        assert handler.can_handle("/api/v1/sme/teams/workspaces")
        assert handler.can_handle("/api/v1/sme/teams/subscribe")
        assert handler.can_handle("/api/v1/sme/teams/subscriptions")

    def test_can_handle_parameterized_routes(self, handler):
        """Test can_handle for parameterized routes."""
        assert handler.can_handle("/api/v1/sme/teams/workspaces/tenant-123")
        assert handler.can_handle("/api/v1/sme/teams/workspaces/tenant-123/test")
        assert handler.can_handle("/api/v1/sme/teams/channels/tenant-123")
        assert handler.can_handle("/api/v1/sme/teams/subscriptions/sub-123")

    def test_cannot_handle_invalid_routes(self, handler):
        """Test can_handle returns False for invalid routes."""
        assert not handler.can_handle("/api/v1/sme/slack/workspaces")
        assert not handler.can_handle("/api/v1/teams/workspaces")
        assert not handler.can_handle("/invalid")


class TestListWorkspaces:
    """Tests for listing workspaces."""

    def test_list_workspaces_empty(self, handler, mock_workspace_store):
        """Test listing workspaces when none exist."""
        mock_workspace_store.get_by_aragora_tenant.return_value = []

        request = MockRequest(command="GET")
        result = handler._list_workspaces(request, {}, user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["workspaces"] == []
        assert body["total"] == 0

    def test_list_workspaces_with_results(self, handler, mock_workspace_store):
        """Test listing workspaces with results."""
        workspaces = [
            MockTeamsWorkspace(tenant_id="tenant-1"),
            MockTeamsWorkspace(tenant_id="tenant-2"),
        ]
        mock_workspace_store.get_by_aragora_tenant.return_value = workspaces

        request = MockRequest(command="GET")
        result = handler._list_workspaces(request, {}, user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["workspaces"]) == 2
        assert body["total"] == 2

    def test_list_workspaces_pagination(self, handler, mock_workspace_store):
        """Test listing workspaces with pagination."""
        workspaces = [MockTeamsWorkspace(tenant_id=f"tenant-{i}") for i in range(10)]
        mock_workspace_store.get_by_aragora_tenant.return_value = workspaces

        # Mock query params
        request = MockRequest(command="GET")
        request.headers = {"limit": "5", "offset": "3"}

        with patch("aragora.server.handlers.sme.teams_workspace.get_string_param") as mock_param:
            mock_param.side_effect = lambda h, k, d: {"limit": "5", "offset": "3"}.get(k, d)
            result = handler._list_workspaces(request, {}, user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["workspaces"]) == 5
        assert body["total"] == 10
        assert body["offset"] == 3


class TestGetWorkspace:
    """Tests for getting workspace details."""

    def test_get_workspace_not_found(self, handler, mock_workspace_store):
        """Test getting non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(command="GET")
        result = handler._get_workspace(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 404

    def test_get_workspace_wrong_org(self, handler, mock_workspace_store):
        """Test getting workspace from different org."""
        workspace = MockTeamsWorkspace(aragora_tenant_id="other-org")
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="GET")
        result = handler._get_workspace(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 404

    def test_get_workspace_success(self, handler, mock_workspace_store):
        """Test getting workspace successfully."""
        workspace = MockTeamsWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="GET")
        result = handler._get_workspace(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["workspace"]["tenant_id"] == "tenant-123"


class TestTestConnection:
    """Tests for testing workspace connection."""

    def test_test_connection_not_found(self, handler, mock_workspace_store):
        """Test connection test for non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(command="POST")
        result = handler._test_connection(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 404

    def test_test_connection_success(self, handler, mock_workspace_store):
        """Test connection test success."""
        workspace = MockTeamsWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="POST")
        result = handler._test_connection(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "connected"
        assert body["token_valid"] is True
        assert "tested_at" in body

    def test_test_connection_invalid_token(self, handler, mock_workspace_store):
        """Test connection test with invalid token."""
        workspace = MockTeamsWorkspace(access_token="short")
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="POST")
        result = handler._test_connection(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "invalid_token"
        assert body["token_valid"] is False

    def test_test_connection_expired_token(self, handler, mock_workspace_store):
        """Test connection test with expired token."""
        # Set token expiry to past
        workspace = MockTeamsWorkspace(
            token_expires_at=datetime.now(timezone.utc).timestamp() - 3600
        )
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="POST")
        result = handler._test_connection(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["token_valid"] is False


class TestDisconnectWorkspace:
    """Tests for disconnecting workspaces."""

    def test_disconnect_not_found(self, handler, mock_workspace_store):
        """Test disconnecting non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(command="DELETE")
        result = handler._disconnect_workspace(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 404

    def test_disconnect_success(self, handler, mock_workspace_store, mock_subscription_store):
        """Test disconnecting workspace successfully."""
        workspace = MockTeamsWorkspace()
        mock_workspace_store.get.return_value = workspace
        mock_subscription_store.get_by_org.return_value = []

        request = MockRequest(command="DELETE")
        result = handler._disconnect_workspace(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["disconnected"] is True
        mock_workspace_store.deactivate.assert_called_once_with("tenant-123")

    def test_disconnect_deactivates_subscriptions(
        self, handler, mock_workspace_store, mock_subscription_store
    ):
        """Test that disconnecting also deactivates subscriptions."""
        workspace = MockTeamsWorkspace()
        mock_workspace_store.get.return_value = workspace

        # Mock subscription belonging to this workspace
        mock_sub = MagicMock()
        mock_sub.workspace_id = "tenant-123"
        mock_sub.id = "sub-123"
        mock_subscription_store.get_by_org.return_value = [mock_sub]

        request = MockRequest(command="DELETE")
        result = handler._disconnect_workspace(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 200
        mock_subscription_store.deactivate.assert_called_once_with("sub-123")


class TestListChannels:
    """Tests for listing channels."""

    def test_list_channels_workspace_not_found(self, handler, mock_workspace_store):
        """Test listing channels for non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(command="GET")
        result = handler._list_channels(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 404

    def test_list_channels_success(self, handler, mock_workspace_store):
        """Test listing channels successfully."""
        workspace = MockTeamsWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="GET")
        result = handler._list_channels(request, {}, "tenant-123", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "channels" in body
        assert body["tenant_id"] == "tenant-123"


class TestSubscribeChannel:
    """Tests for subscribing channels."""

    def test_subscribe_missing_fields(self, handler, mock_workspace_store):
        """Test subscription with missing required fields."""
        request = MockRequest(
            command="POST",
            body=json.dumps({}).encode(),
        )
        result = handler._subscribe_channel(request, {}, user=MockUser())

        assert result.status_code == 400

    def test_subscribe_workspace_not_found(
        self, handler, mock_workspace_store, mock_subscription_store
    ):
        """Test subscription to non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(
            command="POST",
            body=json.dumps({"tenant_id": "tenant-123", "channel_id": "channel-123"}).encode(),
        )
        result = handler._subscribe_channel(request, {}, user=MockUser())

        assert result.status_code == 404

    def test_subscribe_invalid_event_type(
        self, handler, mock_workspace_store, mock_subscription_store
    ):
        """Test subscription with invalid event type."""
        workspace = MockTeamsWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(
            command="POST",
            body=json.dumps(
                {
                    "tenant_id": "tenant-123",
                    "channel_id": "channel-123",
                    "event_types": ["invalid_event"],
                }
            ).encode(),
        )
        result = handler._subscribe_channel(request, {}, user=MockUser())

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "invalid event type" in body.get("error", "").lower()

    def test_subscribe_success(self, handler, mock_workspace_store, mock_subscription_store):
        """Test successful subscription."""
        workspace = MockTeamsWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(
            command="POST",
            body=json.dumps(
                {
                    "tenant_id": "tenant-123",
                    "channel_id": "channel-123",
                    "channel_name": "General",
                    "event_types": ["receipt", "budget_alert"],
                }
            ).encode(),
        )
        result = handler._subscribe_channel(request, {}, user=MockUser())

        assert result.status_code == 201
        body = json.loads(result.body)
        assert "subscription" in body
        mock_subscription_store.create.assert_called_once()

    def test_subscribe_duplicate(self, handler, mock_workspace_store, mock_subscription_store):
        """Test subscription conflict."""
        workspace = MockTeamsWorkspace()
        mock_workspace_store.get.return_value = workspace
        mock_subscription_store.create.side_effect = ValueError("Subscription already exists")

        request = MockRequest(
            command="POST",
            body=json.dumps(
                {
                    "tenant_id": "tenant-123",
                    "channel_id": "channel-123",
                    "event_types": ["receipt"],
                }
            ).encode(),
        )
        result = handler._subscribe_channel(request, {}, user=MockUser())

        assert result.status_code == 409


class TestListSubscriptions:
    """Tests for listing subscriptions."""

    def test_list_subscriptions_empty(self, handler, mock_subscription_store):
        """Test listing subscriptions when none exist."""
        mock_subscription_store.get_by_org.return_value = []

        request = MockRequest(command="GET")
        result = handler._list_subscriptions(request, {}, user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["subscriptions"] == []
        assert body["total"] == 0

    def test_list_subscriptions_with_results(self, handler, mock_subscription_store):
        """Test listing subscriptions with results."""
        mock_sub = MagicMock()
        mock_sub.to_dict.return_value = {
            "id": "sub-123",
            "channel_id": "channel-123",
        }
        mock_subscription_store.get_by_org.return_value = [mock_sub]

        request = MockRequest(command="GET")
        result = handler._list_subscriptions(request, {}, user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["subscriptions"]) == 1
        assert body["total"] == 1


class TestDeleteSubscription:
    """Tests for deleting subscriptions."""

    def test_delete_subscription_not_found(self, handler, mock_subscription_store):
        """Test deleting non-existent subscription."""
        mock_subscription_store.get.return_value = None

        request = MockRequest(command="DELETE")
        result = handler._delete_subscription(request, {}, "sub-123", user=MockUser())

        assert result.status_code == 404

    def test_delete_subscription_wrong_org(self, handler, mock_subscription_store):
        """Test deleting subscription from different org."""
        mock_sub = MagicMock()
        mock_sub.org_id = "other-org"
        mock_subscription_store.get.return_value = mock_sub

        request = MockRequest(command="DELETE")
        result = handler._delete_subscription(request, {}, "sub-123", user=MockUser())

        assert result.status_code == 404

    def test_delete_subscription_success(self, handler, mock_subscription_store):
        """Test deleting subscription successfully."""
        mock_sub = MagicMock()
        mock_sub.org_id = "org-456"
        mock_subscription_store.get.return_value = mock_sub

        request = MockRequest(command="DELETE")
        result = handler._delete_subscription(request, {}, "sub-123", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deleted"] is True
        mock_subscription_store.delete.assert_called_once_with("sub-123")


class TestRouteMatching:
    """Tests for route matching."""

    def test_match_route_workspace_detail(self, handler):
        """Test matching workspace detail route."""
        route_name, param_id = handler._match_route("/api/v1/sme/teams/workspaces/tenant-123")
        assert route_name == "workspace_detail"
        assert param_id == "tenant-123"

    def test_match_route_workspace_test(self, handler):
        """Test matching workspace test route."""
        route_name, param_id = handler._match_route("/api/v1/sme/teams/workspaces/tenant-123/test")
        assert route_name == "workspace_test"
        assert param_id == "tenant-123"

    def test_match_route_channels(self, handler):
        """Test matching channels route."""
        route_name, param_id = handler._match_route("/api/v1/sme/teams/channels/tenant-123")
        assert route_name == "channels"
        assert param_id == "tenant-123"

    def test_match_route_subscription_detail(self, handler):
        """Test matching subscription detail route."""
        route_name, param_id = handler._match_route("/api/v1/sme/teams/subscriptions/sub-123")
        assert route_name == "subscription_detail"
        assert param_id == "sub-123"

    def test_match_route_no_match(self, handler):
        """Test no match for invalid route."""
        route_name, param_id = handler._match_route("/api/v1/invalid")
        assert route_name is None
        assert param_id is None


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_allows_normal_traffic(self, handler, mock_ctx):
        """Test rate limiter allows normal traffic."""
        from aragora.server.handlers.sme.teams_workspace import _teams_limiter

        # Clear any existing state
        _teams_limiter._buckets.clear()

        request = MockRequest(command="GET")
        # Should allow up to 30 requests per minute
        for _ in range(5):
            result = handler.handle("/api/v1/sme/teams/workspaces", {}, request, method="GET")
            # Will fail auth but not rate limit
            assert result.status_code != 429
