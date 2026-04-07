"""Tests for aragora.server.handlers.social.social_media - Social Media Handler."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .conftest import MockHandler, install_social_slack_stubs

install_social_slack_stubs()

from aragora.server.handlers.social.social_media import SocialMediaHandler


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
    social_handler_context_builder,
    mock_user_store,
    mock_youtube_connector,
    mock_twitter_connector,
    mock_storage,
    mock_audio_store,
):
    return social_handler_context_builder(
        user_store=mock_user_store,
        youtube_connector=mock_youtube_connector,
        twitter_connector=mock_twitter_connector,
        storage=mock_storage,
        audio_store=mock_audio_store,
    )


@pytest.fixture
def social_handler(handler_context):
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
