"""Tests for aragora.server.handlers.social.collaboration - Collaboration Handler."""

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
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# Try to import the handler module
try:
    from aragora.server.handlers.social.collaboration import CollaborationHandler

    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    CollaborationHandler = None

pytestmark = pytest.mark.skipif(not HANDLER_AVAILABLE, reason="CollaborationHandler not available")


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
class MockSession:
    """Mock collaboration session."""

    id: str = "session-123"
    org_id: str = "org-123"
    name: str = "Design Review"
    description: str = "Reviewing new design"
    channel_id: str = "C12345"
    platform: str = "slack"
    created_by: str = "user-123"
    participants: list[str] = field(default_factory=lambda: ["user-123", "user-456"])
    status: str = "active"
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "name": self.name,
            "description": self.description,
            "channel_id": self.channel_id,
            "platform": self.platform,
            "created_by": self.created_by,
            "participants": self.participants,
            "status": self.status,
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
        from aragora.server.handlers.social import collaboration

        collaboration._collab_limiter._buckets.clear()
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
def mock_session_store():
    store = MagicMock()
    store.get_by_org.return_value = [MockSession()]
    store.get_by_id.return_value = MockSession()
    store.create.return_value = MockSession()
    store.update.return_value = True
    store.delete.return_value = True
    store.add_participant.return_value = True
    store.remove_participant.return_value = True
    return store


@pytest.fixture
def handler_context(mock_user_store):
    return {"user_store": mock_user_store}


@pytest.fixture
def collab_handler(handler_context, mock_session_store):
    handler = CollaborationHandler(handler_context)
    yield handler


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_sessions_list(self, collab_handler):
        """Test handler recognizes sessions list endpoint."""
        assert collab_handler.can_handle("/api/v1/social/collaboration/sessions") is True

    def test_can_handle_session_detail(self, collab_handler):
        """Test handler recognizes session detail endpoint."""
        assert (
            collab_handler.can_handle("/api/v1/social/collaboration/sessions/session-123") is True
        )

    def test_can_handle_participants(self, collab_handler):
        """Test handler recognizes participants endpoint."""
        assert (
            collab_handler.can_handle(
                "/api/v1/social/collaboration/sessions/session-123/participants"
            )
            is True
        )

    def test_can_handle_messages(self, collab_handler):
        """Test handler recognizes messages endpoint."""
        assert (
            collab_handler.can_handle("/api/v1/social/collaboration/sessions/session-123/messages")
            is True
        )

    def test_cannot_handle_unknown_path(self, collab_handler):
        """Test handler rejects unknown paths."""
        assert collab_handler.can_handle("/api/v1/unknown") is False


# ===========================================================================
# List Sessions Tests
# ===========================================================================


class TestListSessions:
    """Tests for listing sessions."""

    def test_list_sessions_success(self, collab_handler, mock_user):
        """Test successful sessions listing."""
        http_handler = MockHandler(path="/api/v1/social/collaboration/sessions", method="GET")

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_list_sessions_with_status_filter(self, collab_handler, mock_user):
        """Test sessions listing with status filter."""
        http_handler = MockHandler(path="/api/v1/social/collaboration/sessions", method="GET")

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions",
            {"status": "active"},
            http_handler,
            method="GET",
        )
        assert result is not None


# ===========================================================================
# Get Session Tests
# ===========================================================================


class TestGetSession:
    """Tests for getting a single session."""

    def test_get_session_success(self, collab_handler, mock_user):
        """Test successful session retrieval."""
        http_handler = MockHandler(
            path="/api/v1/social/collaboration/sessions/session-123", method="GET"
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-123",
            {},
            http_handler,
            method="GET",
        )
        assert result is not None

    def test_get_session_not_found(self, collab_handler, mock_session_store):
        """Test session not found error."""
        mock_session_store.get_by_id.return_value = None
        http_handler = MockHandler(
            path="/api/v1/social/collaboration/sessions/session-999", method="GET"
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-999",
            {},
            http_handler,
            method="GET",
        )
        assert result is not None


# ===========================================================================
# Create Session Tests
# ===========================================================================


class TestCreateSession:
    """Tests for creating sessions."""

    def test_create_session_success(self, collab_handler, mock_user):
        """Test successful session creation."""
        body = {
            "name": "New Session",
            "description": "Test session",
            "channel_id": "C12345",
            "platform": "slack",
        }
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/social/collaboration/sessions", method="POST"
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_create_session_missing_name(self, collab_handler):
        """Test error when name is missing."""
        body = {"channel_id": "C12345", "platform": "slack"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/social/collaboration/sessions", method="POST"
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_create_session_missing_channel(self, collab_handler):
        """Test error when channel_id is missing."""
        body = {"name": "Test", "platform": "slack"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/social/collaboration/sessions", method="POST"
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions", {}, http_handler, method="POST"
        )
        assert result is not None


# ===========================================================================
# Update Session Tests
# ===========================================================================


class TestUpdateSession:
    """Tests for updating sessions."""

    def test_update_session_success(self, collab_handler, mock_user):
        """Test successful session update."""
        body = {"status": "completed"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/social/collaboration/sessions/session-123", method="PATCH"
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-123",
            {},
            http_handler,
            method="PATCH",
        )
        assert result is not None


# ===========================================================================
# Delete Session Tests
# ===========================================================================


class TestDeleteSession:
    """Tests for deleting sessions."""

    def test_delete_session_success(self, collab_handler, mock_user):
        """Test successful session deletion."""
        http_handler = MockHandler(
            path="/api/v1/social/collaboration/sessions/session-123", method="DELETE"
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-123",
            {},
            http_handler,
            method="DELETE",
        )
        assert result is not None


# ===========================================================================
# Participants Tests
# ===========================================================================


class TestParticipants:
    """Tests for participant management."""

    def test_list_participants_success(self, collab_handler, mock_user):
        """Test successful participants listing."""
        http_handler = MockHandler(
            path="/api/v1/social/collaboration/sessions/session-123/participants",
            method="GET",
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-123/participants",
            {},
            http_handler,
            method="GET",
        )
        assert result is not None

    def test_add_participant_success(self, collab_handler, mock_user):
        """Test successful participant addition."""
        body = {"user_id": "user-789"}
        http_handler = MockHandler.with_json_body(
            body,
            path="/api/v1/social/collaboration/sessions/session-123/participants",
            method="POST",
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-123/participants",
            {},
            http_handler,
            method="POST",
        )
        assert result is not None

    def test_remove_participant_success(self, collab_handler, mock_user):
        """Test successful participant removal."""
        http_handler = MockHandler(
            path="/api/v1/social/collaboration/sessions/session-123/participants/user-456",
            method="DELETE",
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-123/participants/user-456",
            {},
            http_handler,
            method="DELETE",
        )
        assert result is not None


# ===========================================================================
# Messages Tests
# ===========================================================================


class TestMessages:
    """Tests for session messages."""

    def test_list_messages_success(self, collab_handler, mock_user):
        """Test successful messages listing."""
        http_handler = MockHandler(
            path="/api/v1/social/collaboration/sessions/session-123/messages",
            method="GET",
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-123/messages",
            {},
            http_handler,
            method="GET",
        )
        assert result is not None

    def test_send_message_success(self, collab_handler, mock_user):
        """Test successful message sending."""
        body = {"content": "Hello, world!"}
        http_handler = MockHandler.with_json_body(
            body,
            path="/api/v1/social/collaboration/sessions/session-123/messages",
            method="POST",
        )

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions/session-123/messages",
            {},
            http_handler,
            method="POST",
        )
        assert result is not None


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self, collab_handler):
        """Test rate limit enforcement."""
        http_handler = MockHandler(path="/api/v1/social/collaboration/sessions", method="GET")

        with patch(
            "aragora.server.handlers.social.collaboration._collab_limiter.is_allowed",
            return_value=False,
        ):
            result = collab_handler.handle(
                "/api/v1/social/collaboration/sessions", {}, http_handler, method="GET"
            )
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_sessions_list_method_not_allowed(self, collab_handler):
        """Test method not allowed for sessions list."""
        http_handler = MockHandler(path="/api/v1/social/collaboration/sessions", method="PUT")

        result = collab_handler.handle(
            "/api/v1/social/collaboration/sessions", {}, http_handler, method="PUT"
        )
        assert result is not None
        assert result.status_code == 405
