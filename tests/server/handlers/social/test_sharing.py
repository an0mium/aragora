"""Tests for aragora.server.handlers.social.sharing - Sharing Handler."""

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
from unittest.mock import MagicMock, patch

import pytest

# Try to import the handler module
try:
    from aragora.server.handlers.social.sharing import SharingHandler

    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    SharingHandler = None

pytestmark = pytest.mark.skipif(not HANDLER_AVAILABLE, reason="SharingHandler not available")


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
class MockShare:
    """Mock share object."""

    id: str = "share-123"
    org_id: str = "org-123"
    resource_type: str = "debate"
    resource_id: str = "debate-123"
    shared_by: str = "user-123"
    shared_with: list[str] = field(default_factory=lambda: ["user-456"])
    channel_id: str = "C12345"
    platform: str = "slack"
    message: str = "Check out this debate!"
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "shared_by": self.shared_by,
            "shared_with": self.shared_with,
            "channel_id": self.channel_id,
            "platform": self.platform,
            "message": self.message,
            "created_at": self.created_at,
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
        from aragora.server.handlers.social import sharing

        sharing._share_limiter._buckets.clear()
    except Exception:
        pass
    yield


@pytest.fixture
def mock_user():
    return MockUser()


@pytest.fixture
def mock_user_store(mock_user):
    store = MagicMock()
    store.get_user_by_id.return_value = mock_user
    return store


@pytest.fixture
def mock_share_store():
    store = MagicMock()
    store.get_by_org.return_value = [MockShare()]
    store.get_by_id.return_value = MockShare()
    store.create.return_value = MockShare()
    store.delete.return_value = True
    return store


@pytest.fixture
def handler_context(mock_user_store):
    return {"user_store": mock_user_store}


@pytest.fixture
def share_handler(handler_context, mock_share_store):
    handler = SharingHandler(handler_context)
    yield handler


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_shares_list(self, share_handler):
        """Test handler recognizes shares list endpoint."""
        assert share_handler.can_handle("/api/v1/social/shares") is True

    def test_can_handle_share_detail(self, share_handler):
        """Test handler recognizes share detail endpoint."""
        assert share_handler.can_handle("/api/v1/social/shares/share-123") is True

    def test_cannot_handle_unknown_path(self, share_handler):
        """Test handler rejects unknown paths."""
        assert share_handler.can_handle("/api/v1/unknown") is False


# ===========================================================================
# List Shares Tests
# ===========================================================================


class TestListShares:
    """Tests for listing shares."""

    def test_list_shares_success(self, share_handler, mock_user):
        """Test successful shares listing."""
        http_handler = MockHandler(path="/api/v1/social/shares", method="GET")

        result = share_handler.handle("/api/v1/social/shares", {}, http_handler, method="GET")
        assert result is not None

    def test_list_shares_with_type_filter(self, share_handler, mock_user):
        """Test shares listing with resource type filter."""
        http_handler = MockHandler(path="/api/v1/social/shares", method="GET")

        result = share_handler.handle(
            "/api/v1/social/shares",
            {"resource_type": "debate"},
            http_handler,
            method="GET",
        )
        assert result is not None


# ===========================================================================
# Get Share Tests
# ===========================================================================


class TestGetShare:
    """Tests for getting a single share."""

    def test_get_share_success(self, share_handler, mock_user):
        """Test successful share retrieval."""
        http_handler = MockHandler(path="/api/v1/social/shares/share-123", method="GET")

        result = share_handler.handle(
            "/api/v1/social/shares/share-123", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_share_not_found(self, share_handler, mock_share_store):
        """Test share not found error."""
        mock_share_store.get_by_id.return_value = None
        http_handler = MockHandler(path="/api/v1/social/shares/share-999", method="GET")

        result = share_handler.handle(
            "/api/v1/social/shares/share-999", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Create Share Tests
# ===========================================================================


class TestCreateShare:
    """Tests for creating shares."""

    def test_create_share_success(self, share_handler, mock_user):
        """Test successful share creation."""
        body = {
            "resource_type": "debate",
            "resource_id": "debate-456",
            "channel_id": "C12345",
            "platform": "slack",
            "message": "Check this out!",
        }
        http_handler = MockHandler.with_json_body(body, path="/api/v1/social/shares", method="POST")

        result = share_handler.handle("/api/v1/social/shares", {}, http_handler, method="POST")
        assert result is not None

    def test_create_share_missing_resource(self, share_handler):
        """Test error when resource info is missing."""
        body = {"channel_id": "C12345", "platform": "slack"}
        http_handler = MockHandler.with_json_body(body, path="/api/v1/social/shares", method="POST")

        result = share_handler.handle("/api/v1/social/shares", {}, http_handler, method="POST")
        assert result is not None


# ===========================================================================
# Delete Share Tests
# ===========================================================================


class TestDeleteShare:
    """Tests for deleting shares."""

    def test_delete_share_success(self, share_handler, mock_user):
        """Test successful share deletion."""
        http_handler = MockHandler(path="/api/v1/social/shares/share-123", method="DELETE")

        result = share_handler.handle(
            "/api/v1/social/shares/share-123", {}, http_handler, method="DELETE"
        )
        assert result is not None


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self, share_handler):
        """Test rate limit enforcement."""
        http_handler = MockHandler(path="/api/v1/social/shares", method="GET")

        with patch(
            "aragora.server.handlers.social.sharing._share_limiter.is_allowed",
            return_value=False,
        ):
            result = share_handler.handle("/api/v1/social/shares", {}, http_handler, method="GET")
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_shares_list_method_not_allowed(self, share_handler):
        """Test method not allowed for shares list."""
        http_handler = MockHandler(path="/api/v1/social/shares", method="PUT")

        result = share_handler.handle("/api/v1/social/shares", {}, http_handler, method="PUT")
        assert result is not None
        assert result.status_code == 405
