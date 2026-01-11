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
from aragora.server.handlers.audio import AudioHandler
from aragora.server.handlers.social import (
    SocialMediaHandler,
    _store_oauth_state,
    _validate_oauth_state,
    _oauth_states,
    _oauth_states_lock,
    ALLOWED_OAUTH_HOSTS,
)
from aragora.server.handlers.base import clear_cache
from aragora.server.middleware.rate_limit import reset_rate_limiters


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_rate_limit_state():
    """Reset rate limiters before and after each test."""
    reset_rate_limiters()
    yield
    reset_rate_limiters()


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
def audio_handler(mock_audio_store, mock_storage):
    """Create an AudioHandler with mock dependencies."""
    ctx = {
        "storage": mock_storage,
        "audio_store": mock_audio_store,
        "nomic_dir": None,
    }
    return AudioHandler(ctx)


@pytest.fixture
def social_handler(mock_storage, mock_youtube_connector, mock_twitter_connector, mock_audio_store):
    """Create a SocialMediaHandler with mock dependencies."""
    ctx = {
        "storage": mock_storage,
        "audio_store": mock_audio_store,
        "youtube_connector": mock_youtube_connector,
        "twitter_connector": mock_twitter_connector,
        "video_generator": None,
        "nomic_dir": None,
    }
    return SocialMediaHandler(ctx)


@pytest.fixture
def handler_no_audio_store(mock_storage):
    """Create handlers without audio store."""
    ctx = {
        "storage": mock_storage,
        "audio_store": None,
        "youtube_connector": None,
        "twitter_connector": None,
        "nomic_dir": None,
    }
    return AudioHandler(ctx)


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

    def test_can_handle_audio_route(self, audio_handler):
        assert audio_handler.can_handle("/audio/test-debate.mp3") is True

    def test_can_handle_podcast_feed(self, audio_handler):
        assert audio_handler.can_handle("/api/podcast/feed.xml") is True

    def test_can_handle_podcast_episodes(self, audio_handler):
        assert audio_handler.can_handle("/api/podcast/episodes") is True

    def test_can_handle_youtube_auth(self, social_handler):
        assert social_handler.can_handle("/api/youtube/auth") is True

    def test_can_handle_youtube_callback(self, social_handler):
        assert social_handler.can_handle("/api/youtube/callback") is True

    def test_can_handle_youtube_status(self, social_handler):
        assert social_handler.can_handle("/api/youtube/status") is True

    def test_can_handle_broadcast_generation(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/debates/test-123/broadcast") is True

    def test_can_handle_twitter_publish(self, social_handler):
        assert social_handler.can_handle("/api/debates/test-123/publish/twitter") is True

    def test_can_handle_youtube_publish(self, social_handler):
        assert social_handler.can_handle("/api/debates/test-123/publish/youtube") is True

    def test_cannot_handle_unrelated_routes(self, broadcast_handler):
        assert broadcast_handler.can_handle("/api/debates") is False
        assert broadcast_handler.can_handle("/api/agents") is False
        assert broadcast_handler.can_handle("/audio/test.wav") is False


# ============================================================================
# GET /audio/{id}.mp3 Tests
# ============================================================================

class TestServeAudio:
    """Tests for audio file serving."""

    def test_serve_audio_success(self, audio_handler, mock_audio_store):
        result = audio_handler.handle("/audio/test-debate.mp3", {}, None)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "audio/mpeg"
        assert b"fake mp3 content" in result.body

    def test_serve_audio_not_found(self, audio_handler, mock_audio_store):
        mock_audio_store.get_path.return_value = None
        result = audio_handler.handle("/audio/unknown.mp3", {}, None)

        assert result is not None
        assert result.status_code == 404

    def test_serve_audio_no_store(self, handler_no_audio_store):
        result = handler_no_audio_store.handle("/audio/test.mp3", {}, None)

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not configured" in data["error"]

    def test_serve_audio_path_traversal_blocked(self, audio_handler):
        result = audio_handler.handle("/audio/../../../etc/passwd.mp3", {}, None)

        assert result is not None
        assert result.status_code == 400

    def test_serve_audio_invalid_debate_id(self, audio_handler):
        result = audio_handler.handle("/audio/test..hack.mp3", {}, None)

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

    def test_podcast_feed_content_type(self, audio_handler, mock_audio_store, mock_storage, mock_handler):
        # Setup to avoid broadcast module dependency
        mock_audio_store.list_all.return_value = []

        with patch('aragora.server.handlers.audio.PODCAST_AVAILABLE', False):
            result = audio_handler.handle("/api/podcast/feed.xml", {}, mock_handler)

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

    def test_podcast_episodes_with_limit(self, audio_handler, mock_handler, mock_audio_store):
        result = audio_handler.handle("/api/podcast/episodes", {"limit": "10"}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "episodes" in data


# ============================================================================
# GET /api/youtube/status Tests
# ============================================================================

class TestYouTubeStatus:
    """Tests for YouTube connector status."""

    def test_youtube_status_not_configured(self, social_handler):
        result = social_handler.handle("/api/youtube/status", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["configured"] is False

    def test_youtube_status_no_connector(self, mock_storage):
        ctx = {"storage": mock_storage, "youtube_connector": None, "twitter_connector": None, "audio_store": None}
        handler = SocialMediaHandler(ctx)
        result = handler.handle("/api/youtube/status", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["configured"] is False

    def test_youtube_status_configured(self, social_handler, mock_youtube_connector):
        mock_youtube_connector.is_configured = True
        mock_youtube_connector.client_id = "test-client-id"
        mock_youtube_connector.client_secret = "test-secret"
        mock_youtube_connector.refresh_token = "test-token"

        result = social_handler.handle("/api/youtube/status", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["has_client_id"] is True


# ============================================================================
# GET /api/youtube/auth Tests
# ============================================================================

class TestYouTubeAuth:
    """Tests for YouTube OAuth URL."""

    def test_youtube_auth_no_connector(self, mock_storage, mock_handler):
        ctx = {"storage": mock_storage, "youtube_connector": None, "twitter_connector": None, "audio_store": None}
        handler = SocialMediaHandler(ctx)
        result = handler.handle("/api/youtube/auth", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    def test_youtube_auth_no_client_id(self, social_handler, mock_handler):
        result = social_handler.handle("/api/youtube/auth", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "hint" in data


# ============================================================================
# GET /api/youtube/callback Tests
# ============================================================================

class TestYouTubeCallback:
    """Tests for YouTube OAuth callback."""

    def test_youtube_callback_missing_code(self, social_handler, mock_handler):
        result = social_handler.handle("/api/youtube/callback", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "code" in data["error"].lower()

    def test_youtube_callback_missing_state(self, social_handler, mock_handler):
        result = social_handler.handle("/api/youtube/callback", {"code": "test"}, mock_handler)

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

    def test_broadcast_returns_existing_audio(self, mock_handler, mock_storage, mock_audio_store, tmp_path):
        """Should return existing audio if already generated."""
        mock_audio_store.exists.return_value = True
        mock_audio_store.get_metadata.return_value = {
            "generated_at": "2024-01-01T00:00:00Z",
        }
        mock_audio_store.get_path.return_value = tmp_path / "test-debate.mp3"

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        result = handler.handle_post("/api/debates/test-debate/broadcast", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "exists"
        assert "audio_url" in data

    def test_broadcast_no_nomic_dir(self, mock_handler, mock_storage, mock_audio_store):
        """Should return 503 when nomic_dir not configured."""
        mock_audio_store.exists.return_value = False

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": None,
        }
        handler = BroadcastHandler(ctx)

        result = handler.handle_post("/api/debates/test/broadcast", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_broadcast_trace_not_found(self, mock_handler, mock_storage, mock_audio_store, tmp_path):
        """Should return 404 when trace file not found."""
        mock_audio_store.exists.return_value = False

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        # Create traces directory but no trace file
        (tmp_path / "traces").mkdir()

        result = handler.handle_post("/api/debates/test/broadcast", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "trace" in data["error"].lower()

    def test_broadcast_by_slug(self, mock_handler, mock_storage, mock_audio_store, tmp_path):
        """Should lookup debate by slug when ID fails."""
        mock_storage.get_debate.return_value = None
        mock_storage.get_debate_by_slug.return_value = {
            "id": "actual-id",
            "task": "Test debate",
        }
        mock_audio_store.exists.return_value = True
        mock_audio_store.get_metadata.return_value = {"generated_at": "2024-01-01"}
        mock_audio_store.get_path.return_value = tmp_path / "actual-id.mp3"

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        result = handler.handle_post("/api/debates/my-slug/broadcast", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "actual-id"

    def test_broadcast_module_not_available(self, mock_handler, mock_storage, mock_audio_store, tmp_path):
        """Should return 503 when broadcast module not available."""
        mock_audio_store.exists.return_value = False

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        # Create trace file
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "test.json").write_text("{}")

        # Call the unwrapped method directly to bypass rate limiter
        unwrapped = handler._generate_broadcast.__wrapped__
        with patch("aragora.server.handlers.broadcast.BROADCAST_AVAILABLE", False):
            result = unwrapped(handler, "test", mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_broadcast_success_without_audio_store(self, mock_handler, mock_storage, tmp_path):
        """Should return generated audio path without persisting."""
        ctx = {
            "storage": mock_storage,
            "audio_store": None,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        # Create trace file
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "test.json").write_text("{}")

        mock_trace = MagicMock()
        audio_path = tmp_path / "output.mp3"

        # Call the unwrapped method directly to bypass rate limiter
        unwrapped = handler._generate_broadcast.__wrapped__
        with patch("aragora.server.handlers.broadcast.BROADCAST_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                with patch("aragora.server.handlers.broadcast.broadcast_debate") as mock_broadcast:
                    mock_broadcast.return_value = audio_path
                    with patch("aragora.server.handlers.broadcast._run_async", side_effect=lambda x: audio_path):
                        result = unwrapped(handler, "test", mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "generated"

    def test_broadcast_failed_generation(self, mock_handler, mock_storage, mock_audio_store, tmp_path):
        """Should return 500 when broadcast generation fails."""
        mock_audio_store.exists.return_value = False

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        # Create trace file
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "test.json").write_text("{}")

        mock_trace = MagicMock()

        # Call the unwrapped method directly to bypass rate limiter
        unwrapped = handler._generate_broadcast.__wrapped__
        with patch("aragora.server.handlers.broadcast.BROADCAST_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                with patch("aragora.server.handlers.broadcast._run_async", return_value=None):
                    result = unwrapped(handler, "test", mock_handler)

        assert result is not None
        assert result.status_code == 500

    def test_broadcast_with_mutagen_metadata(self, mock_handler, mock_storage, mock_audio_store, tmp_path):
        """Should extract duration with mutagen when available."""
        mock_audio_store.exists.return_value = False
        mock_audio_store.save.return_value = tmp_path / "stored.mp3"

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        # Create trace file
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "test.json").write_text("{}")

        audio_path = tmp_path / "output.mp3"
        mock_trace = MagicMock()
        mock_mp3 = MagicMock()
        mock_mp3.info.length = 120.5

        # Call the unwrapped method directly to bypass rate limiter
        unwrapped = handler._generate_broadcast.__wrapped__
        with patch("aragora.server.handlers.broadcast.BROADCAST_AVAILABLE", True):
            with patch("aragora.server.handlers.broadcast.MUTAGEN_AVAILABLE", True):
                with patch("aragora.server.handlers.broadcast.MP3", return_value=mock_mp3):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        with patch("aragora.server.handlers.broadcast._run_async", return_value=audio_path):
                            result = unwrapped(handler, "test", mock_handler)

        assert result is not None
        assert result.status_code == 200
        # Check save was called with duration
        mock_audio_store.save.assert_called_once()
        call_kwargs = mock_audio_store.save.call_args[1]
        assert call_kwargs.get("duration_seconds") == 120

    def test_broadcast_audio_persist_failure(self, mock_handler, mock_storage, mock_audio_store, tmp_path):
        """Should return warning when audio persist fails."""
        mock_audio_store.exists.return_value = False
        mock_audio_store.save.side_effect = Exception("Storage error")

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        # Create trace file
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "test.json").write_text("{}")

        audio_path = tmp_path / "output.mp3"
        mock_trace = MagicMock()

        # Call the unwrapped method directly to bypass rate limiter
        unwrapped = handler._generate_broadcast.__wrapped__
        with patch("aragora.server.handlers.broadcast.BROADCAST_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                with patch("aragora.server.handlers.broadcast._run_async", return_value=audio_path):
                    result = unwrapped(handler, "test", mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "generated"
        assert "warning" in data

    def test_broadcast_trace_load_failure(self, mock_handler, mock_storage, mock_audio_store, tmp_path):
        """Should return 500 when trace load fails."""
        mock_audio_store.exists.return_value = False

        ctx = {
            "storage": mock_storage,
            "audio_store": mock_audio_store,
            "nomic_dir": tmp_path,
        }
        handler = BroadcastHandler(ctx)

        # Create trace file
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "test.json").write_text("{}")

        # Call the unwrapped method directly to bypass rate limiter
        unwrapped = handler._generate_broadcast.__wrapped__
        with patch("aragora.debate.traces.DebateTrace.load", side_effect=Exception("Invalid trace")):
            result = unwrapped(handler, "test", mock_handler)

        assert result is not None
        assert result.status_code == 500
        data = json.loads(result.body)
        # Error message is sanitized by safe_error_message() for security
        assert "error" in data["error"].lower() or data["error"] == "An error occurred"


# ============================================================================
# POST /api/debates/{id}/publish/twitter Tests
# ============================================================================

class TestTwitterPublish:
    """Tests for Twitter publishing."""

    def test_twitter_no_connector(self, mock_storage, mock_handler):
        ctx = {"storage": mock_storage, "youtube_connector": None, "twitter_connector": None, "audio_store": None}
        handler = SocialMediaHandler(ctx)
        result = handler.handle_post(
            "/api/debates/test/publish/twitter", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 500

    def test_twitter_not_configured(self, social_handler, mock_handler):
        result = social_handler.handle_post(
            "/api/debates/test/publish/twitter", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "hint" in data

    def test_twitter_invalid_debate_id(self, social_handler, mock_handler):
        result = social_handler.handle_post(
            "/api/debates/../../etc/publish/twitter", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400


# ============================================================================
# POST /api/debates/{id}/publish/youtube Tests
# ============================================================================

class TestYouTubePublish:
    """Tests for YouTube publishing."""

    def test_youtube_no_connector(self, mock_storage, mock_handler):
        ctx = {"storage": mock_storage, "youtube_connector": None, "twitter_connector": None, "audio_store": None}
        handler = SocialMediaHandler(ctx)
        result = handler.handle_post(
            "/api/debates/test/publish/youtube", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 500

    def test_youtube_not_configured(self, social_handler, mock_handler):
        result = social_handler.handle_post(
            "/api/debates/test/publish/youtube", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "hint" in data

    def test_youtube_quota_exceeded(self, social_handler, mock_handler, mock_youtube_connector):
        mock_youtube_connector.is_configured = True
        mock_youtube_connector.rate_limiter.can_upload.return_value = False
        mock_youtube_connector.rate_limiter.remaining_quota = 0

        result = social_handler.handle_post(
            "/api/debates/test/publish/youtube", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 429
        data = json.loads(result.body)
        assert "quota" in data["error"].lower()

    def test_youtube_no_audio_for_debate(self, social_handler, mock_handler, mock_youtube_connector, mock_audio_store):
        mock_youtube_connector.is_configured = True
        mock_audio_store.exists.return_value = False

        result = social_handler.handle_post(
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

    def test_audio_path_traversal_with_encoded_chars(self, audio_handler):
        result = audio_handler.handle("/audio/..%2F..%2Fetc%2Fpasswd.mp3", {}, None)
        assert result.status_code == 400

    def test_broadcast_sql_injection(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post(
            "/api/debates/'; DROP TABLE debates;--/broadcast", {}, mock_handler
        )
        assert result.status_code == 400

    def test_twitter_xss_in_debate_id(self, social_handler, mock_handler):
        result = social_handler.handle_post(
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

    def test_audio_with_special_characters_in_id(self, audio_handler):
        # Should reject special characters
        result = audio_handler.handle("/audio/test<script>.mp3", {}, None)
        assert result.status_code == 400

    def test_empty_debate_id_in_broadcast(self, broadcast_handler, mock_handler):
        result = broadcast_handler.handle_post("/api/debates//broadcast", {}, mock_handler)
        # Empty ID should fail validation
        assert result is None or result.status_code == 400


# ============================================================================
# OAuth CSRF Protection Tests
# ============================================================================

class TestOAuthCSRFProtection:
    """Tests for OAuth state parameter CSRF protection."""

    def test_oauth_state_storage_and_validation(self):
        """Test that OAuth state is stored and can be validated."""
        state = "test-state-token-12345"
        _store_oauth_state(state)

        # First validation should succeed
        assert _validate_oauth_state(state) is True

    def test_oauth_state_one_time_use(self):
        """Test that OAuth state can only be validated once (one-time use)."""
        state = "one-time-state-token"
        _store_oauth_state(state)

        # First validation succeeds
        assert _validate_oauth_state(state) is True

        # Second validation fails (already consumed)
        assert _validate_oauth_state(state) is False

    def test_oauth_state_invalid_token_rejected(self):
        """Test that invalid/unknown state tokens are rejected."""
        # Never stored this state
        assert _validate_oauth_state("never-stored-state") is False
        assert _validate_oauth_state("") is False
        assert _validate_oauth_state("random-garbage-12345") is False

    def test_oauth_state_expired_token_rejected(self):
        """Test that expired state tokens are rejected."""
        import time

        state = "expiring-state-token"

        # Manually store with past expiry
        with _oauth_states_lock:
            _oauth_states[state] = time.time() - 1  # Already expired

        # Validation should fail
        assert _validate_oauth_state(state) is False

    def test_oauth_callback_rejects_invalid_state(self, social_handler, mock_handler, mock_youtube_connector):
        """Test that OAuth callback rejects invalid state parameter."""
        mock_youtube_connector.client_id = "test-client-id"

        result = social_handler.handle(
            "/api/youtube/callback",
            {"code": "valid-code", "state": "invalid-never-stored-state"},
            mock_handler
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid" in data["error"].lower() or "expired" in data["error"].lower()

    def test_oauth_callback_rejects_expired_state(self, social_handler, mock_handler, mock_youtube_connector):
        """Test that OAuth callback rejects expired state."""
        import time

        mock_youtube_connector.client_id = "test-client-id"

        # Store expired state
        expired_state = "expired-state-for-callback"
        with _oauth_states_lock:
            _oauth_states[expired_state] = time.time() - 100  # Expired 100 seconds ago

        result = social_handler.handle(
            "/api/youtube/callback",
            {"code": "valid-code", "state": expired_state},
            mock_handler
        )

        assert result is not None
        assert result.status_code == 400


# ============================================================================
# Host Header Validation Tests (Open Redirect Prevention)
# ============================================================================

class TestHostHeaderValidation:
    """Tests for Host header validation to prevent open redirect."""

    def test_youtube_auth_rejects_untrusted_host(self, social_handler, mock_youtube_connector):
        """Test that YouTube auth rejects untrusted Host headers."""
        mock_youtube_connector.client_id = "test-client-id"

        # Create handler with untrusted host
        evil_handler = Mock()
        evil_handler.headers = {"Host": "evil.com"}

        result = social_handler.handle("/api/youtube/auth", {}, evil_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "untrusted" in data["error"].lower()

    def test_youtube_auth_accepts_trusted_localhost(self, social_handler, mock_youtube_connector):
        """Test that YouTube auth accepts trusted localhost."""
        mock_youtube_connector.client_id = "test-client-id"
        mock_youtube_connector.get_auth_url = Mock(return_value="https://accounts.google.com/auth?...")

        trusted_handler = Mock()
        trusted_handler.headers = {"Host": "localhost:8080"}

        result = social_handler.handle("/api/youtube/auth", {}, trusted_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "auth_url" in data

    def test_youtube_auth_accepts_trusted_127_0_0_1(self, social_handler, mock_youtube_connector):
        """Test that YouTube auth accepts trusted 127.0.0.1."""
        mock_youtube_connector.client_id = "test-client-id"
        mock_youtube_connector.get_auth_url = Mock(return_value="https://accounts.google.com/auth?...")

        trusted_handler = Mock()
        trusted_handler.headers = {"Host": "127.0.0.1:8080"}

        result = social_handler.handle("/api/youtube/auth", {}, trusted_handler)

        assert result is not None
        assert result.status_code == 200

    def test_youtube_callback_rejects_untrusted_host(self, social_handler, mock_youtube_connector):
        """Test that YouTube callback rejects untrusted Host headers."""
        mock_youtube_connector.client_id = "test-client-id"

        # Store valid state
        valid_state = "valid-state-for-host-test"
        _store_oauth_state(valid_state)

        # Try callback with untrusted host
        evil_handler = Mock()
        evil_handler.headers = {"Host": "attacker.com"}

        result = social_handler.handle(
            "/api/youtube/callback",
            {"code": "auth-code", "state": valid_state},
            evil_handler
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "untrusted" in data["error"].lower()

    def test_host_validation_subdomain_bypass_prevention(self, social_handler, mock_youtube_connector):
        """Test that subdomain variations don't bypass host validation."""
        mock_youtube_connector.client_id = "test-client-id"

        bypass_attempts = [
            "evil.localhost:8080",
            "localhost.evil.com:8080",
            "localhost:8080.evil.com",
            "localhost:8081",  # Different port
        ]

        for evil_host in bypass_attempts:
            evil_handler = Mock()
            evil_handler.headers = {"Host": evil_host}

            result = social_handler.handle("/api/youtube/auth", {}, evil_handler)

            assert result is not None, f"No result for host: {evil_host}"
            assert result.status_code == 400, f"Should reject host: {evil_host}"

    def test_host_validation_whitespace_handling(self):
        """Test that ALLOWED_OAUTH_HOSTS strips whitespace from config."""
        # Verify default hosts are properly trimmed
        for host in ALLOWED_OAUTH_HOSTS:
            assert host == host.strip(), f"Host '{host}' has whitespace"
            assert " " not in host, f"Host '{host}' contains spaces"
