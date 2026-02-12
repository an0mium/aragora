"""Tests for Slack Workspace Handler."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.sme.slack_workspace import SlackWorkspaceHandler


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
class MockSlackWorkspace:
    """Mock Slack workspace for testing."""

    workspace_id: str = "T12345678"
    workspace_name: str = "Test Workspace"
    access_token: str = "xoxb-test-token-12345"
    bot_user_id: str = "U12345678"
    installed_at: float = 1700000000.0
    installed_by: str | None = "user-123"
    scopes: list[str] = None
    tenant_id: str = "org-456"
    is_active: bool = True
    signing_secret: str | None = None
    refresh_token: str | None = None
    token_expires_at: float | None = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = ["chat:write", "channels:read"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "bot_user_id": self.bot_user_id,
            "installed_at": self.installed_at,
            "installed_at_iso": datetime.fromtimestamp(
                self.installed_at, tz=timezone.utc
            ).isoformat(),
            "installed_by": self.installed_by,
            "scopes": self.scopes,
            "tenant_id": self.tenant_id,
            "is_active": self.is_active,
            "has_signing_secret": bool(self.signing_secret),
            "has_refresh_token": bool(self.refresh_token),
            "token_expires_at": self.token_expires_at,
        }


class MockRequest(dict):
    """Mock HTTP request handler that also acts as a dict for query params."""

    def __init__(
        self,
        command: str = "GET",
        path: str = "/",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        query_params: dict[str, str] | None = None,
    ):
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
    store.get_by_tenant.return_value = []
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
            "channel_type": "slack",
            "channel_id": "C12345678",
        },
    )
    store.delete.return_value = True
    store.deactivate.return_value = True
    return store


@pytest.fixture
def handler(mock_ctx, mock_workspace_store, mock_subscription_store):
    """Create handler with mocked dependencies."""
    h = SlackWorkspaceHandler(mock_ctx)

    # Patch store methods
    h._get_workspace_store = MagicMock(return_value=mock_workspace_store)
    h._get_subscription_store = MagicMock(return_value=mock_subscription_store)

    return h


class TestSlackWorkspaceHandler:
    """Tests for SlackWorkspaceHandler."""

    def test_can_handle_static_routes(self, handler):
        """Test can_handle for static routes."""
        assert handler.can_handle("/api/v1/sme/slack/workspaces")
        assert handler.can_handle("/api/v1/sme/slack/subscribe")
        assert handler.can_handle("/api/v1/sme/slack/subscriptions")

    def test_can_handle_parameterized_routes(self, handler):
        """Test can_handle for parameterized routes."""
        assert handler.can_handle("/api/v1/sme/slack/workspaces/T12345678")
        assert handler.can_handle("/api/v1/sme/slack/workspaces/T12345678/test")
        assert handler.can_handle("/api/v1/sme/slack/channels/T12345678")
        assert handler.can_handle("/api/v1/sme/slack/subscriptions/sub-123")

    def test_cannot_handle_invalid_routes(self, handler):
        """Test can_handle returns False for invalid routes."""
        assert not handler.can_handle("/api/v1/sme/teams/workspaces")
        assert not handler.can_handle("/api/v1/slack/workspaces")
        assert not handler.can_handle("/invalid")


class TestListWorkspaces:
    """Tests for listing workspaces."""

    def test_list_workspaces_empty(self, handler, mock_workspace_store):
        """Test listing workspaces when none exist."""
        mock_workspace_store.get_by_tenant.return_value = []

        request = MockRequest(command="GET")
        result = handler._list_workspaces(request, {}, user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["workspaces"] == []
        assert body["total"] == 0

    def test_list_workspaces_with_results(self, handler, mock_workspace_store):
        """Test listing workspaces with results."""
        workspaces = [
            MockSlackWorkspace(workspace_id="T1"),
            MockSlackWorkspace(workspace_id="T2"),
        ]
        mock_workspace_store.get_by_tenant.return_value = workspaces

        request = MockRequest(command="GET")
        result = handler._list_workspaces(request, {}, user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["workspaces"]) == 2
        assert body["total"] == 2


class TestGetWorkspace:
    """Tests for getting workspace details."""

    def test_get_workspace_not_found(self, handler, mock_workspace_store):
        """Test getting non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(command="GET")
        result = handler._get_workspace(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 404

    def test_get_workspace_wrong_org(self, handler, mock_workspace_store):
        """Test getting workspace from different org."""
        workspace = MockSlackWorkspace(tenant_id="other-org")
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="GET")
        result = handler._get_workspace(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 404

    def test_get_workspace_success(self, handler, mock_workspace_store):
        """Test getting workspace successfully."""
        workspace = MockSlackWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="GET")
        result = handler._get_workspace(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["workspace"]["workspace_id"] == "T12345678"


class TestTestConnection:
    """Tests for testing workspace connection."""

    def test_test_connection_not_found(self, handler, mock_workspace_store):
        """Test connection test for non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(command="POST")
        result = handler._test_connection(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 404

    def test_test_connection_success(self, handler, mock_workspace_store):
        """Test connection test success."""
        workspace = MockSlackWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="POST")
        result = handler._test_connection(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "connected"
        assert body["token_valid"] is True
        assert "tested_at" in body

    def test_test_connection_invalid_token(self, handler, mock_workspace_store):
        """Test connection test with invalid token."""
        workspace = MockSlackWorkspace(access_token="invalid-token")
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="POST")
        result = handler._test_connection(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "invalid_token"
        assert body["token_valid"] is False

    def test_test_connection_expired_token(self, handler, mock_workspace_store):
        """Test connection test with expired token."""
        workspace = MockSlackWorkspace(
            token_expires_at=datetime.now(timezone.utc).timestamp() - 3600
        )
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="POST")
        result = handler._test_connection(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["token_valid"] is False


class TestDisconnectWorkspace:
    """Tests for disconnecting workspaces."""

    def test_disconnect_not_found(self, handler, mock_workspace_store):
        """Test disconnecting non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(command="DELETE")
        result = handler._disconnect_workspace(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 404

    def test_disconnect_success(self, handler, mock_workspace_store, mock_subscription_store):
        """Test disconnecting workspace successfully."""
        workspace = MockSlackWorkspace()
        mock_workspace_store.get.return_value = workspace
        mock_subscription_store.get_by_org.return_value = []

        request = MockRequest(command="DELETE")
        result = handler._disconnect_workspace(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["disconnected"] is True
        mock_workspace_store.deactivate.assert_called_once_with("T12345678")


class TestListChannels:
    """Tests for listing channels."""

    def test_list_channels_workspace_not_found(self, handler, mock_workspace_store):
        """Test listing channels for non-existent workspace."""
        mock_workspace_store.get.return_value = None

        request = MockRequest(command="GET")
        result = handler._list_channels(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 404

    def test_list_channels_success(self, handler, mock_workspace_store):
        """Test listing channels successfully."""
        workspace = MockSlackWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(command="GET")
        result = handler._list_channels(request, {}, "T12345678", user=MockUser())

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "channels" in body
        assert body["workspace_id"] == "T12345678"


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
            body=json.dumps({"workspace_id": "T12345678", "channel_id": "C12345678"}).encode(),
        )
        result = handler._subscribe_channel(request, {}, user=MockUser())

        assert result.status_code == 404

    def test_subscribe_invalid_event_type(
        self, handler, mock_workspace_store, mock_subscription_store
    ):
        """Test subscription with invalid event type."""
        workspace = MockSlackWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(
            command="POST",
            body=json.dumps(
                {
                    "workspace_id": "T12345678",
                    "channel_id": "C12345678",
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
        workspace = MockSlackWorkspace()
        mock_workspace_store.get.return_value = workspace

        request = MockRequest(
            command="POST",
            body=json.dumps(
                {
                    "workspace_id": "T12345678",
                    "channel_id": "C12345678",
                    "channel_name": "#general",
                    "event_types": ["receipt", "budget_alert"],
                }
            ).encode(),
        )
        result = handler._subscribe_channel(request, {}, user=MockUser())

        assert result.status_code == 201
        body = json.loads(result.body)
        assert "subscription" in body
        mock_subscription_store.create.assert_called_once()


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
            "channel_id": "C12345678",
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
        route_name, param_id = handler._match_route("/api/v1/sme/slack/workspaces/T12345678")
        assert route_name == "workspace_detail"
        assert param_id == "T12345678"

    def test_match_route_workspace_test(self, handler):
        """Test matching workspace test route."""
        route_name, param_id = handler._match_route("/api/v1/sme/slack/workspaces/T12345678/test")
        assert route_name == "workspace_test"
        assert param_id == "T12345678"

    def test_match_route_channels(self, handler):
        """Test matching channels route."""
        route_name, param_id = handler._match_route("/api/v1/sme/slack/channels/T12345678")
        assert route_name == "channels"
        assert param_id == "T12345678"

    def test_match_route_subscription_detail(self, handler):
        """Test matching subscription detail route."""
        route_name, param_id = handler._match_route("/api/v1/sme/slack/subscriptions/sub-123")
        assert route_name == "subscription_detail"
        assert param_id == "sub-123"

    def test_match_route_no_match(self, handler):
        """Test no match for invalid route."""
        route_name, param_id = handler._match_route("/api/v1/invalid")
        assert route_name is None
        assert param_id is None
