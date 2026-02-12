"""Tests for aragora.server.handlers.social.social_media - Social Media Handler."""

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
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Try to import the handler module
try:
    from aragora.server.handlers.social.social_media import SocialMediaHandler

    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    SocialMediaHandler = None

pytestmark = pytest.mark.skipif(not HANDLER_AVAILABLE, reason="SocialMediaHandler not available")


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
def reset_oauth_state():
    """Reset OAuth state before each test."""
    try:
        from aragora.server.handlers.social import social_media

        social_media._oauth_states.clear()
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
def mock_youtube_connector():
    connector = MagicMock()
    connector.is_configured = True
    connector.client_id = "test-client-id"
    connector.client_secret = "test-secret"
    connector.refresh_token = "test-refresh-token"
    connector.rate_limiter = MagicMock()
    connector.rate_limiter.remaining_quota = 10000
    connector.rate_limiter.can_upload.return_value = True
    connector.circuit_breaker = MagicMock()
    connector.circuit_breaker.is_open = False
    connector.get_auth_url.return_value = "https://accounts.google.com/oauth"
    connector.exchange_code = AsyncMock(return_value={"success": True})
    connector.upload = AsyncMock(
        return_value={
            "success": True,
            "video_id": "abc123",
            "url": "https://youtube.com/watch?v=abc123",
        }
    )
    return connector


@pytest.fixture
def mock_twitter_connector():
    connector = MagicMock()
    connector.is_configured = True
    connector.post_tweet = AsyncMock(
        return_value={
            "success": True,
            "tweet_id": "12345",
            "url": "https://twitter.com/i/status/12345",
        }
    )
    connector.post_thread = AsyncMock(return_value={"success": True, "thread_ids": ["1", "2", "3"]})
    return connector


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.get_debate.return_value = {
        "id": "debate-123",
        "task": "Test debate topic",
        "agents": ["claude", "gpt-4"],
        "verdict": "Consensus reached",
    }
    storage.get_debate_by_slug.return_value = storage.get_debate.return_value
    return storage


@pytest.fixture
def mock_audio_store():
    store = MagicMock()
    store.exists.return_value = True
    store.get_path.return_value = Path("/tmp/debate-123.mp3")
    return store


@pytest.fixture
def handler_context(
    mock_user_store, mock_youtube_connector, mock_twitter_connector, mock_storage, mock_audio_store
):
    return {
        "user_store": mock_user_store,
        "youtube_connector": mock_youtube_connector,
        "twitter_connector": mock_twitter_connector,
        "storage": mock_storage,
        "audio_store": mock_audio_store,
    }


@pytest.fixture
def social_handler(handler_context):
    if not HANDLER_AVAILABLE:
        pytest.skip("SocialMediaHandler not available")
    handler = SocialMediaHandler(handler_context)
    yield handler


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_youtube_auth(self, social_handler):
        """Test handler recognizes YouTube auth endpoint."""
        assert social_handler.can_handle("/api/v1/youtube/auth") is True

    def test_can_handle_youtube_callback(self, social_handler):
        """Test handler recognizes YouTube callback endpoint."""
        assert social_handler.can_handle("/api/v1/youtube/callback") is True

    def test_can_handle_youtube_status(self, social_handler):
        """Test handler recognizes YouTube status endpoint."""
        assert social_handler.can_handle("/api/v1/youtube/status") is True

    def test_can_handle_twitter_publish(self, social_handler):
        """Test handler recognizes Twitter publish endpoint."""
        assert social_handler.can_handle("/api/v1/debates/debate-123/publish/twitter") is True

    def test_can_handle_youtube_publish(self, social_handler):
        """Test handler recognizes YouTube publish endpoint."""
        assert social_handler.can_handle("/api/v1/debates/debate-123/publish/youtube") is True

    def test_cannot_handle_unknown_path(self, social_handler):
        """Test handler rejects unknown paths."""
        assert social_handler.can_handle("/api/v1/unknown") is False


# ===========================================================================
# YouTube Status Tests
# ===========================================================================


class TestYouTubeStatus:
    """Tests for YouTube status endpoint."""

    def test_youtube_status_configured(self, social_handler, mock_user):
        """Test YouTube status when configured."""
        http_handler = MockHandler(path="/api/v1/youtube/status", method="GET")

        result = social_handler.handle("/api/v1/youtube/status", {}, http_handler, method="GET")
        assert result is not None

    def test_youtube_status_not_configured(self, handler_context):
        """Test YouTube status when not configured."""
        handler_context["youtube_connector"] = None
        from aragora.server.handlers.social.social_media import SocialMediaHandler

        handler = SocialMediaHandler(handler_context)
        http_handler = MockHandler(path="/api/v1/youtube/status", method="GET")

        result = handler.handle("/api/v1/youtube/status", {}, http_handler, method="GET")
        assert result is not None


# ===========================================================================
# YouTube OAuth Tests
# ===========================================================================


class TestYouTubeOAuth:
    """Tests for YouTube OAuth endpoints."""

    def test_youtube_auth_missing_client_id(self, handler_context, mock_youtube_connector):
        """Test YouTube auth without client ID."""
        mock_youtube_connector.client_id = ""
        from aragora.server.handlers.social.social_media import SocialMediaHandler

        handler = SocialMediaHandler(handler_context)
        http_handler = MockHandler(
            path="/api/v1/youtube/auth",
            method="GET",
            headers={"Host": "localhost:8080"},
        )

        result = handler.handle("/api/v1/youtube/auth", {}, http_handler, method="GET")
        assert result is not None

    def test_youtube_callback_missing_code(self, social_handler):
        """Test YouTube callback without authorization code."""
        http_handler = MockHandler(
            path="/api/v1/youtube/callback",
            method="GET",
            headers={"Host": "localhost:8080"},
        )

        result = social_handler.handle(
            "/api/v1/youtube/callback", {"state": "test-state"}, http_handler, method="GET"
        )
        assert result is not None

    def test_youtube_callback_missing_state(self, social_handler):
        """Test YouTube callback without state parameter."""
        http_handler = MockHandler(
            path="/api/v1/youtube/callback",
            method="GET",
            headers={"Host": "localhost:8080"},
        )

        result = social_handler.handle(
            "/api/v1/youtube/callback", {"code": "test-code"}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_youtube_status_get_only(self, social_handler):
        """Test only GET allowed for YouTube status."""
        http_handler = MockHandler(path="/api/v1/youtube/status", method="POST")

        result = social_handler.handle("/api/v1/youtube/status", {}, http_handler, method="POST")
        # POST is not handled by handle(), so it falls through to handle_post
        # which doesn't match, returning None
        assert result is None
