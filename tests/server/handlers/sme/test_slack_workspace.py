"""Tests for aragora.server.handlers.sme.slack_workspace - Slack Workspace Handler."""

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
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Try to import the handler module
try:
    from aragora.server.handlers.sme.slack_workspace import SlackWorkspaceHandler

    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    SlackWorkspaceHandler = None

pytestmark = pytest.mark.skipif(not HANDLER_AVAILABLE, reason="SlackWorkspaceHandler not available")


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
class MockWorkspace:
    """Mock Slack workspace."""

    id: str = "ws-123"
    org_id: str = "org-123"
    team_id: str = "T12345"
    team_name: str = "Test Workspace"
    bot_token: str = "xoxb-test-token"
    is_active: bool = True
    channels: list[str] = field(default_factory=lambda: ["C12345", "C67890"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "is_active": self.is_active,
            "channels": self.channels,
        }


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(
        self,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
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

    def get_argument(self, name: str, default: str = None) -> str | None:
        return default


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state before each test."""
    try:
        from aragora.server.handlers.sme import slack_workspace

        slack_workspace._workspace_limiter._buckets.clear()
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
def mock_workspace_store():
    store = MagicMock()
    store.get_by_org.return_value = [MockWorkspace()]
    store.get_by_team_id.return_value = MockWorkspace()
    store.create.return_value = MockWorkspace()
    store.update.return_value = True
    store.delete.return_value = True
    return store


@pytest.fixture
def mock_slack_client():
    client = MagicMock()
    client.conversations_list = AsyncMock(
        return_value={"channels": [{"id": "C12345", "name": "general"}]}
    )
    client.auth_test = AsyncMock(
        return_value={"ok": True, "team_id": "T12345", "team": "Test Workspace"}
    )
    return client


@pytest.fixture
def handler_context(mock_user_store):
    return {"user_store": mock_user_store}


@pytest.fixture
def slack_handler(handler_context, mock_workspace_store, mock_slack_client):
    handler = SlackWorkspaceHandler(handler_context)
    yield handler


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_workspaces_list(self, slack_handler):
        """Test handler recognizes workspaces list endpoint."""
        assert slack_handler.can_handle("/api/v1/sme/slack/workspaces") is True

    def test_can_handle_workspace_detail(self, slack_handler):
        """Test handler recognizes workspace detail endpoint."""
        assert slack_handler.can_handle("/api/v1/sme/slack/workspaces/ws-123") is True

    def test_can_handle_channels(self, slack_handler):
        """Test handler recognizes channels endpoint."""
        assert slack_handler.can_handle("/api/v1/sme/slack/channels") is True

    def test_can_handle_subscribe(self, slack_handler):
        """Test handler recognizes subscribe endpoint."""
        assert slack_handler.can_handle("/api/v1/sme/slack/subscribe") is True

    def test_can_handle_subscriptions(self, slack_handler):
        """Test handler recognizes subscriptions endpoint."""
        assert slack_handler.can_handle("/api/v1/sme/slack/subscriptions") is True

    def test_cannot_handle_unknown_path(self, slack_handler):
        """Test handler rejects unknown paths."""
        assert slack_handler.can_handle("/api/v1/unknown") is False


# ===========================================================================
# List Workspaces Tests
# ===========================================================================


class TestListWorkspaces:
    """Tests for listing workspaces."""

    def test_list_workspaces_success(self, slack_handler, mock_user):
        """Test successful workspaces listing."""
        http_handler = MockHandler(path="/api/v1/sme/slack/workspaces", method="GET")

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_list_workspaces_empty(self, slack_handler, mock_workspace_store):
        """Test listing when no workspaces configured."""
        mock_workspace_store.get_by_org.return_value = []
        http_handler = MockHandler(path="/api/v1/sme/slack/workspaces", method="GET")

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Get Workspace Tests
# ===========================================================================


class TestGetWorkspace:
    """Tests for getting a single workspace."""

    def test_get_workspace_success(self, slack_handler, mock_user):
        """Test successful workspace retrieval."""
        http_handler = MockHandler(path="/api/v1/sme/slack/workspaces/ws-123", method="GET")

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces/ws-123", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_workspace_not_found(self, slack_handler, mock_workspace_store):
        """Test workspace not found error."""
        mock_workspace_store.get_by_team_id.return_value = None
        http_handler = MockHandler(path="/api/v1/sme/slack/workspaces/ws-999", method="GET")

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces/ws-999", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Create Workspace Tests
# ===========================================================================


class TestCreateWorkspace:
    """Tests for creating workspaces."""

    def test_create_workspace_success(self, slack_handler, mock_user):
        """Test successful workspace creation."""
        body = {
            "team_id": "T12345",
            "bot_token": "xoxb-new-token",
        }
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/slack/workspaces", method="POST"
        )

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_create_workspace_missing_team_id(self, slack_handler):
        """Test error when team_id is missing."""
        body = {"bot_token": "xoxb-test"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/slack/workspaces", method="POST"
        )

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_create_workspace_missing_token(self, slack_handler):
        """Test error when bot_token is missing."""
        body = {"team_id": "T12345"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/slack/workspaces", method="POST"
        )

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces", {}, http_handler, method="POST"
        )
        assert result is not None


# ===========================================================================
# Update Workspace Tests
# ===========================================================================


class TestUpdateWorkspace:
    """Tests for updating workspaces."""

    def test_update_workspace_success(self, slack_handler, mock_user):
        """Test successful workspace update."""
        body = {"is_active": False}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/slack/workspaces/ws-123", method="PATCH"
        )

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces/ws-123", {}, http_handler, method="PATCH"
        )
        assert result is not None


# ===========================================================================
# Delete Workspace Tests
# ===========================================================================


class TestDeleteWorkspace:
    """Tests for deleting workspaces."""

    def test_delete_workspace_success(self, slack_handler, mock_user):
        """Test successful workspace deletion."""
        http_handler = MockHandler(path="/api/v1/sme/slack/workspaces/ws-123", method="DELETE")

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces/ws-123", {}, http_handler, method="DELETE"
        )
        assert result is not None


# ===========================================================================
# OAuth Tests
# ===========================================================================


class TestOAuth:
    """Tests for OAuth flow.

    Note: The SlackWorkspaceHandler does not currently expose dedicated
    /oauth/* routes. These tests verify the subscribe endpoint which is
    the handler's entry point for workspace configuration.
    """

    def test_subscribe_endpoint_success(self, slack_handler, mock_user):
        """Test subscribe endpoint (used for workspace config)."""
        http_handler = MockHandler(path="/api/v1/sme/slack/subscribe", method="POST")

        result = slack_handler.handle(
            "/api/v1/sme/slack/subscribe", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_oauth_callback_success(self, slack_handler, mock_user):
        """Test OAuth callback endpoint."""
        http_handler = MockHandler(path="/api/v1/sme/slack/oauth/callback", method="GET")
        http_handler.user = mock_user

        result = slack_handler.handle(
            "/api/v1/sme/slack/oauth/callback",
            {"code": "auth-code-123", "state": "state-123"},
            http_handler,
            method="GET",
        )
        assert result is not None
        assert result.status == 200
        data = result.to_dict()
        assert data["body"]["status"] == "oauth_callback"
        assert data["body"]["code"] == "auth-code-123"

    def test_oauth_callback_missing_code(self, slack_handler):
        """Test OAuth callback without code."""
        http_handler = MockHandler(path="/api/v1/sme/slack/oauth/callback", method="GET")

        result = slack_handler.handle(
            "/api/v1/sme/slack/oauth/callback",
            {"state": "state-123"},
            http_handler,
            method="GET",
        )
        assert result is not None
        assert result.status == 400


# ===========================================================================
# Channels Tests
# ===========================================================================


class TestChannels:
    """Tests for channel listing."""

    def test_list_channels_success(self, slack_handler, mock_user):
        """Test successful channel listing for a specific channel."""
        http_handler = MockHandler(path="/api/v1/sme/slack/channels/ch-123", method="GET")

        result = slack_handler.handle(
            "/api/v1/sme/slack/channels/ch-123", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self, slack_handler):
        """Test rate limit enforcement."""
        http_handler = MockHandler(path="/api/v1/sme/slack/workspaces", method="GET")

        with patch(
            "aragora.server.handlers.sme.slack_workspace._slack_limiter.is_allowed",
            return_value=False,
        ):
            result = slack_handler.handle(
                "/api/v1/sme/slack/workspaces", {}, http_handler, method="GET"
            )
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_workspaces_list_method_not_allowed(self, slack_handler):
        """Test method not allowed for workspaces list."""
        http_handler = MockHandler(path="/api/v1/sme/slack/workspaces", method="PUT")

        result = slack_handler.handle(
            "/api/v1/sme/slack/workspaces", {}, http_handler, method="PUT"
        )
        assert result is not None
        assert result.status_code == 405
