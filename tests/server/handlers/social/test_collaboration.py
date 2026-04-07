"""Tests for aragora.server.handlers.social.collaboration - Collaboration Handler."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from .conftest import MockHandler, install_social_slack_stubs

install_social_slack_stubs()

from aragora.server.handlers.social.collaboration import CollaborationHandler


# ===========================================================================
# Mock Classes
# ===========================================================================


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
