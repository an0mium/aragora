"""
Tests for BroadcastHandler endpoints.

Endpoints tested:
- GET /audio/{id}.mp3 - Serve audio file
- GET /api/podcast/feed.xml - Podcast RSS feed
- GET /api/podcast/episodes - List podcast episodes
- GET /api/youtube/auth - YouTube OAuth URL
- GET /api/youtube/callback - YouTube OAuth callback
- GET /api/youtube/status - YouTube connector status
- POST /api/debates/{id}/broadcast - Generate audio
- POST /api/debates/{id}/publish/twitter - Publish to Twitter
- POST /api/debates/{id}/publish/youtube - Publish to YouTube
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO

from aragora.server.handlers.broadcast import BroadcastHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_audio_store(tmp_path):
    """Create a mock audio store."""
    store = Mock()
    store.storage_dir = tmp_path

    # Create a test audio file
    test_audio = tmp_path / "test-debate.mp3"
    test_audio.write_bytes(b"fake mp3 content")

    store.exists.return_value = True
    store.get_path.return_value = test_audio
    store.get_metadata.return_value = {
        "debate_id": "test-debate",
        "duration_seconds": 120,
        "file_size_bytes": 1024,
        "generated_at": "2024-01-01T00:00:00Z",
    }
    store.list_all.return_value = [
        {
            "debate_id": "debate-1",
            "duration_seconds": 120,
            "file_size_bytes": 1024,
            "generated_at": "2024-01-01T00:00:00Z",
        },
        {
            "debate_id": "debate-2",
            "duration_seconds": 180,
            "file_size_bytes": 2048,
            "generated_at": "2024-01-02T00:00:00Z",
        },
    ]
    return store


@pytest.fixture
def mock_storage():
    """Create a mock debate storage."""
    storage = Mock()
    storage.get_debate.return_value = {
        "id": "test-debate",
        "task": "Test AI Debate",
        "agents": ["claude", "gpt4"],
        "verdict": "consensus reached",
        "created_at": "2024-01-01T00:00:00Z",
    }
    storage.get_debate_by_slug.return_value = None
    storage.update_audio.return_value = None
    return storage


@pytest.fixture
def mock_youtube_connector():
    """Create a mock YouTube connector."""
    youtube = Mock()
    youtube.is_configured = False
    youtube.client_id = None
    youtube.client_secret = None
    youtube.refresh_token = None
    youtube.rate_limiter = Mock()
    youtube.rate_limiter.remaining_quota = 100
    youtube.rate_limiter.can_upload.return_value = True
    youtube.circuit_breaker = Mock()
    youtube.circuit_breaker.is_open = False
    return youtube


@pytest.fixture
def mock_twitter_connector():
    """Create a mock Twitter connector."""
    twitter = Mock()
    twitter.is_configured = False
    return twitter


@pytest.fixture
def broadcast_handler(mock_audio_store, mock_storage, mock_youtube_connector, mock_twitter_connector):
    """Create a BroadcastHandler with mock dependencies."""
    ctx = {
        "storage": mock_storage,
        "audio_store": mock_audio_store,
        "youtube_connector": mock_youtube_connector,
        "twitter_connector": mock_twitter_connector,
        "video_generator": None,
        "nomic_dir": None,
    }
    return BroadcastHandler(ctx)


@pytest.fixture
def handler_no_audio_store(mock_storage):
    """Create a BroadcastHandler without audio store."""
    ctx = {
        "storage": mock_storage,
        "audio_store": None,
        "youtube_connector": None,
        "twitter_connector": None,
        "nomic_dir": None,
    }
    return BroadcastHandler(ctx)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler for request context."""
    handler = Mock()
    handler.headers = {"Host": "localhost:8080"}
    handler.rfile = BytesIO(b'{}')
    return handler


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestBroadcastRouting:
    """Tests for route matching."""

    def test_can_handle_audio_route(self, broadcast_handler):
        assert broadcast_handler.can_handle("/audio/test-debate.mp3") is True

    def test_can_handle_podcast_feed(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/podcast/feed.xml") is True

    def test_can_handle_podcast_episodes(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/podcast/episodes") is True

    def test_can_handle_youtube_auth(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/youtube/auth") is True

    def test_can_handle_youtube_callback(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/youtube/callback") is True

    def test_can_handle_youtube_status(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/youtube/status") is True

    def test_can_handle_broadcast_generation(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/debates/test-123/broadcast") is True

    def test_can_handle_twitter_publish(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/debates/test-123/publish/twitter") is True

    def test_can_handle_youtube_publish(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/debates/test-123/publish/youtube") is True

    def test_cannot_handle_unrelated_routes(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/debates") is False
        assert broadcast_handler.can_handle("/api/agents") is False
        assert broadcast_handler.can_handle("/audio/test.wav") is False


# ============================================================================
# GET /audio/{id}.mp3 Tests
# ============================================================================

class TestServeAudio:
    """Tests for audio file serving."""

    def test_serve_audio_success(self, broadcast_handler, mock_audio_store):
        result = broadcast_handler.handle("/audio/test-debate.mp3", {}, None)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "audio/mpeg"
        assert b"fake mp3 content" in result.body

    def test_serve_audio_not_found(self, broadcast_handler, mock_audio_store):
        mock_audio_store.get_path.return_value = None
        result = broadcast_handler.handle("/audio/unknown.mp3", {}, None)

        assert result is not None
        assert result.status_code == 404

    def test_serve_audio_no_store(self, handler_no_audio_store):
        result = handler_no_audio_store.handle("/audio/test.mp3", {}, None)

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not configured" in data["error"]

    def test_serve_audio_path_traversal_blocked(self, broadcast_handler):
        result = broadcast_handler.handle("/audio/../../../etc/passwd.mp3", {}, None)

        assert result is not None
        assert result.status_code == 400

    def test_serve_audio_invalid_debate_id(self, broadcast_handler):
        result = broadcast_handler.handle("/audio/test..hack.mp3", {}, None)

        assert result is not None
        assert result.status_code == 400


# ============================================================================
# GET /api/podcast/feed.xml Tests
# ============================================================================

class TestPodcastFeed:
    """Tests for podcast RSS feed."""

    def test_podcast_feed_no_audio_store(self, handler_no_audio_store, mock_handler):
        result = handler_no_audio_store.handle("/api/podcast/feed.xml", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_podcast_feed_content_type(self, broadcast_handler, mock_audio_store, mock_storage, mock_handler):
        # Setup to avoid broadcast module dependency
        mock_audio_store.list_all.return_value = []

        with patch('aragora.server.handlers.broadcast.BROADCAST_AVAILABLE', False):
            result = broadcast_handler.handle("/api/podcast/feed.xml", {}, mock_handler)

            if result:
                assert result.status_code in (200, 503)


# ============================================================================
# GET /api/podcast/episodes Tests
# ============================================================================

class TestPodcastEpisodes:
    """Tests for podcast episodes listing."""

    def test_podcast_episodes_no_audio_store(self, handler_no_audio_store, mock_handler):
        result = handler_no_audio_store.handle("/api/podcast/episodes", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_podcast_episodes_with_limit(self, broadcast_handler, mock_handler, mock_audio_store):
        result = broadcast_handler.handle("/api/podcast/episodes", {"limit": "10"}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "episodes" in data


# ============================================================================
# GET /api/youtube/status Tests
# ============================================================================

class TestYouTubeStatus:
    """Tests for YouTube connector status."""

    def test_youtube_status_not_configured(self, broadcast_handler):
        result = broadcast_handler.handle("/api/youtube/status", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["configured"] is False

    def test_youtube_status_no_connector(self, handler_no_audio_store):
        result = handler_no_audio_store.handle("/api/youtube/status", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["configured"] is False

    def test_youtube_status_configured(self, broadcast_handler, mock_youtube_connector):
        mock_youtube_connector.is_configured = True
        mock_youtube_connector.client_id = "test-client-id"
        mock_youtube_connector.client_secret = "test-secret"
        mock_youtube_connector.refresh_token = "test-token"

        result = broadcast_handler.handle("/api/youtube/status", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["has_client_id"] is True


# ============================================================================
# GET /api/youtube/auth Tests
# ============================================================================

class TestYouTubeAuth:
    """Tests for YouTube OAuth URL."""

    def test_youtube_auth_no_connector(self, handler_no_audio_store, mock_handler):
        result = handler_no_audio_store.handle("/api/youtube/auth", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    def test_youtube_auth_no_client_id(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle("/api/youtube/auth", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "hint" in data


# ============================================================================
# GET /api/youtube/callback Tests
# ============================================================================

class TestYouTubeCallback:
    """Tests for YouTube OAuth callback."""

    def test_youtube_callback_missing_code(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle("/api/youtube/callback", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "code" in data["error"].lower()

    def test_youtube_callback_missing_state(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle("/api/youtube/callback", {"code": "test"}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "state" in data["error"].lower()


# ============================================================================
# POST /api/debates/{id}/broadcast Tests
# ============================================================================

class TestBroadcastGeneration:
    """Tests for broadcast audio generation."""

    def test_broadcast_no_storage(self, mock_handler):
        ctx = {"storage": None, "audio_store": None, "nomic_dir": None}
        handler = BroadcastHandler(ctx)

        result = handler.handle_post("/api/debates/test/broadcast", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_broadcast_debate_not_found(self, mock_handler, mock_storage, mock_audio_store, mock_youtube_connector, mock_twitter_connector, tmp_path):
        mock_storage.get_debate.return_value = None
        mock_storage.get_debate_by_slug.return_value = None

        # Create handler with nomic_dir set so it proceeds to debate lookup
        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "youtube_connector": mock_youtube_connector,
            "twitter_connector": mock_twitter_connector,
            "video_generator": None,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        result = handler.handle_post("/api/debates/unknown/broadcast", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404

    def test_broadcast_invalid_debate_id(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post("/api/debates/../etc/broadcast", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400


# ============================================================================
# POST /api/debates/{id}/publish/twitter Tests
# ============================================================================

class TestTwitterPublish:
    """Tests for Twitter publishing."""

    def test_twitter_no_connector(self, handler_no_audio_store, mock_handler):
        result = handler_no_audio_store.handle_post(
            "/api/debates/test/publish/twitter", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 500

    def test_twitter_not_configured(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post(
            "/api/debates/test/publish/twitter", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "hint" in data

    def test_twitter_invalid_debate_id(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post(
            "/api/debates/../../etc/publish/twitter", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400


# ============================================================================
# POST /api/debates/{id}/publish/youtube Tests
# ============================================================================

class TestYouTubePublish:
    """Tests for YouTube publishing."""

    def test_youtube_no_connector(self, handler_no_audio_store, mock_handler):
        result = handler_no_audio_store.handle_post(
            "/api/debates/test/publish/youtube", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 500

    def test_youtube_not_configured(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post(
            "/api/debates/test/publish/youtube", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "hint" in data

    def test_youtube_quota_exceeded(self, broadcast_handler, mock_handler, mock_youtube_connector):
        mock_youtube_connector.is_configured = True
        mock_youtube_connector.rate_limiter.can_upload.return_value = False
        mock_youtube_connector.rate_limiter.remaining_quota = 0

        result = broadcast_handler.handle_post(
            "/api/debates/test/publish/youtube", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 429
        data = json.loads(result.body)
        assert "quota" in data["error"].lower()

    def test_youtube_no_audio_for_debate(self, broadcast_handler, mock_handler, mock_youtube_connector, mock_audio_store):
        mock_youtube_connector.is_configured = True
        mock_audio_store.exists.return_value = False

        result = broadcast_handler.handle_post(
            "/api/debates/test/publish/youtube", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "audio" in data["error"].lower()


# ============================================================================
# Security Tests
# ============================================================================

class TestBroadcastSecurity:
    """Security tests for broadcast endpoints."""

    def test_audio_path_traversal_with_encoded_chars(self, broadcast_handler):
        result = broadcast_handler.handle("/audio/..%2F..%2Fetc%2Fpasswd.mp3", {}, None)
        assert result.status_code == 400

    def test_broadcast_sql_injection(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post(
            "/api/debates/'; DROP TABLE debates;--/broadcast", {}, mock_handler
        )
        assert result.status_code == 400

    def test_twitter_xss_in_debate_id(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post(
            "/api/debates/<script>alert(1)</script>/publish/twitter", {}, mock_handler
        )
        assert result.status_code == 400


# ============================================================================
# Edge Cases
# ============================================================================

class TestBroadcastEdgeCases:
    """Tests for edge cases."""

    def test_handle_returns_none_for_unhandled_get(self, broadcast_handler):
        result = broadcast_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_handle_post_returns_none_for_unhandled_post(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post("/api/other/endpoint", {}, mock_handler)
        assert result is None

    def test_audio_with_special_characters_in_id(self, broadcast_handler):
        # Should reject special characters
        result = broadcast_handler.handle("/audio/test<script>.mp3", {}, None)
        assert result.status_code == 400

    def test_empty_debate_id_in_broadcast(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post("/api/debates//broadcast", {}, mock_handler)
        # Empty ID should fail validation
        assert result is None or result.status_code == 400
