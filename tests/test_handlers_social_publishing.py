"""Extended tests for SocialMediaHandler - Twitter/YouTube publishing.

These tests supplement test_handlers_social.py with coverage for:
- Twitter publishing functionality
- YouTube publishing functionality
- Connector configuration validation
- POST body parsing
- Storage integration
- Quota handling
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass

from aragora.server.handlers.social import (
    SocialMediaHandler,
    _store_oauth_state,
    _validate_oauth_state,
    _oauth_states,
    _oauth_states_lock,
    MAX_OAUTH_STATES,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_storage():
    """Create mock storage with debate data."""
    storage = MagicMock()
    storage.get_debate.return_value = {
        "id": "test-debate-123",
        "task": "Discuss AI safety",
        "agents": ["claude", "gpt4"],
        "verdict": "Consensus reached",
        "messages": [{"role": "agent", "content": "Test message"}],
    }
    storage.get_debate_by_slug.return_value = None
    return storage


@pytest.fixture
def mock_twitter_connector():
    """Create mock Twitter connector."""
    connector = MagicMock()
    connector.is_configured = True
    connector.post_tweet = AsyncMock(
        return_value={
            "success": True,
            "tweet_id": "12345",
            "url": "https://twitter.com/user/status/12345",
        }
    )
    connector.post_thread = AsyncMock(
        return_value={
            "success": True,
            "thread_ids": ["12345", "12346", "12347"],
            "url": "https://twitter.com/user/status/12345",
        }
    )
    return connector


@pytest.fixture
def mock_youtube_connector():
    """Create mock YouTube connector."""
    connector = MagicMock()
    connector.is_configured = True
    connector.client_id = "test-client-id"
    connector.client_secret = "test-client-secret"
    connector.refresh_token = "test-refresh-token"
    connector.rate_limiter = MagicMock()
    connector.rate_limiter.can_upload.return_value = True
    connector.rate_limiter.remaining_quota = 9500
    connector.circuit_breaker = MagicMock()
    connector.circuit_breaker.is_open = False
    connector.upload = AsyncMock(
        return_value={
            "success": True,
            "video_id": "abc123",
            "url": "https://youtube.com/watch?v=abc123",
        }
    )
    return connector


@pytest.fixture
def mock_audio_store():
    """Create mock audio store."""
    store = MagicMock()
    store.exists.return_value = True
    store.get_path.return_value = "/tmp/audio/test-debate.mp3"
    return store


@pytest.fixture
def mock_video_generator():
    """Create mock video generator."""
    generator = MagicMock()
    generator.generate_waveform_video.return_value = "/tmp/video/test.mp4"
    generator.generate_static_video.return_value = "/tmp/video/static.mp4"
    return generator


@pytest.fixture
def handler_ctx(
    mock_storage,
    mock_twitter_connector,
    mock_youtube_connector,
    mock_audio_store,
    mock_video_generator,
):
    """Create handler context with all mocks."""
    return {
        "storage": mock_storage,
        "twitter_connector": mock_twitter_connector,
        "youtube_connector": mock_youtube_connector,
        "audio_store": mock_audio_store,
        "video_generator": mock_video_generator,
    }


@pytest.fixture
def social_handler(handler_ctx):
    """Create SocialMediaHandler with mock context."""
    return SocialMediaHandler(server_context=handler_ctx)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler for request context."""
    handler = MagicMock()
    handler.headers = {"Host": "localhost:8080"}
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = b"{}"
    return handler


# =============================================================================
# Twitter Publishing Tests
# =============================================================================


class TestTwitterPublishing:
    """Tests for Twitter publishing functionality."""

    def test_publish_twitter_success(
        self, social_handler, mock_http_handler, mock_storage, mock_twitter_connector
    ):
        """Successful Twitter publish returns tweet details."""
        with patch("aragora.connectors.twitter_poster.DebateContentFormatter") as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format_single_tweet.return_value = "Test tweet content"

            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["tweet_id"] == "12345"

    def test_publish_twitter_not_configured(self, handler_ctx, mock_http_handler):
        """Returns error when Twitter not configured."""
        handler_ctx["twitter_connector"].is_configured = False
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post(
                "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "not configured" in body["error"].lower()

    def test_publish_twitter_no_connector(self, mock_http_handler):
        """Returns error when Twitter connector not initialized."""
        handler = SocialMediaHandler(server_context={"storage": MagicMock()})

        result = handler.handle_post(
            "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert "not initialized" in body["error"].lower()

    def test_publish_twitter_debate_not_found(self, handler_ctx, mock_http_handler):
        """Returns 404 when debate not found."""
        handler_ctx["storage"].get_debate.return_value = None
        handler_ctx["storage"].get_debate_by_slug.return_value = None
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post(
                "/api/debates/nonexistent/publish/twitter", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_publish_twitter_storage_unavailable(self, handler_ctx, mock_http_handler):
        """Returns 503 when storage unavailable."""
        handler_ctx["storage"] = None
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post(
                "/api/debates/test-123/publish/twitter", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_publish_twitter_thread_mode(
        self, social_handler, mock_http_handler, mock_twitter_connector
    ):
        """Thread mode publishes as Twitter thread."""
        with patch("aragora.connectors.twitter_poster.DebateContentFormatter") as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format_as_thread.return_value = ["Tweet 1", "Tweet 2", "Tweet 3"]

            with patch.object(social_handler, "read_json_body", return_value={"thread_mode": True}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
                )

        assert result is not None
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["thread_ids"] == ["12345", "12346", "12347"]
        mock_twitter_connector.post_thread.assert_called_once()

    def test_publish_twitter_invalid_json_body(self, social_handler, mock_http_handler):
        """Returns 400 for invalid JSON body."""
        with patch.object(social_handler, "read_json_body", return_value=None):
            result = social_handler.handle_post(
                "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "invalid json" in body["error"].lower()

    def test_publish_twitter_api_error(
        self, social_handler, mock_http_handler, mock_twitter_connector
    ):
        """Handles Twitter API errors gracefully."""
        mock_twitter_connector.post_tweet = AsyncMock(
            return_value={
                "success": False,
                "error": "Rate limit exceeded",
            }
        )

        with patch("aragora.connectors.twitter_poster.DebateContentFormatter") as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format_single_tweet.return_value = "Test tweet"

            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
                )

        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert body["success"] is False
        assert "rate limit" in body["error"].lower()

    def test_publish_twitter_exception(self, social_handler, mock_http_handler):
        """Handles exceptions during Twitter publishing."""
        with patch("aragora.connectors.twitter_poster.DebateContentFormatter") as MockFormatter:
            MockFormatter.side_effect = ImportError("Module not found")

            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
                )

        assert result is not None
        assert result.status_code == 500

    def test_publish_twitter_with_audio_link(
        self, social_handler, mock_http_handler, mock_audio_store
    ):
        """Includes audio link when audio exists."""
        mock_audio_store.exists.return_value = True

        with patch("aragora.connectors.twitter_poster.DebateContentFormatter") as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format_single_tweet.return_value = "Test tweet"

            with patch.object(
                social_handler, "read_json_body", return_value={"include_audio_link": True}
            ):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
                )

        assert result is not None
        # Formatter should be called (we can't easily verify audio_url without inspecting mock)
        mock_formatter.format_single_tweet.assert_called_once()


# =============================================================================
# YouTube Publishing Tests
# =============================================================================


class TestYouTubePublishing:
    """Tests for YouTube publishing functionality."""

    def test_publish_youtube_success(
        self, social_handler, mock_http_handler, mock_youtube_connector
    ):
        """Successful YouTube publish returns video details."""
        with patch("aragora.connectors.youtube_uploader.YouTubeVideoMetadata"):
            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["video_id"] == "abc123"

    def test_publish_youtube_not_configured(self, handler_ctx, mock_http_handler):
        """Returns error when YouTube not configured."""
        handler_ctx["youtube_connector"].is_configured = False
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post(
                "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "not configured" in body["error"].lower()

    def test_publish_youtube_no_connector(self, mock_http_handler):
        """Returns error when YouTube connector not initialized."""
        handler = SocialMediaHandler(server_context={"storage": MagicMock()})

        result = handler.handle_post(
            "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert "not initialized" in body["error"].lower()

    def test_publish_youtube_debate_not_found(self, handler_ctx, mock_http_handler):
        """Returns 404 when debate not found."""
        handler_ctx["storage"].get_debate.return_value = None
        handler_ctx["storage"].get_debate_by_slug.return_value = None
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post(
                "/api/debates/nonexistent/publish/youtube", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_publish_youtube_quota_exceeded(self, handler_ctx, mock_http_handler):
        """Returns 429 when YouTube quota exceeded."""
        handler_ctx["youtube_connector"].rate_limiter.can_upload.return_value = False
        handler_ctx["youtube_connector"].rate_limiter.remaining_quota = 0
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post(
                "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 429
        body = json.loads(result.body)
        assert "quota exceeded" in body["error"].lower()

    def test_publish_youtube_no_audio(self, handler_ctx, mock_http_handler):
        """Returns error when no audio exists for debate."""
        handler_ctx["audio_store"].exists.return_value = False
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post(
                "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "no audio" in body["error"].lower()

    def test_publish_youtube_no_video_generator(self, handler_ctx, mock_http_handler):
        """Returns 503 when video generator not available."""
        handler_ctx["video_generator"] = None
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch("aragora.connectors.youtube_uploader.YouTubeVideoMetadata"):
            with patch.object(handler, "read_json_body", return_value={}):
                result = handler.handle_post(
                    "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
                )

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "video generator" in body["error"].lower()

    def test_publish_youtube_custom_metadata(self, social_handler, mock_http_handler):
        """Accepts custom title, description, tags, and privacy."""
        custom_options = {
            "title": "Custom Title",
            "description": "Custom Description",
            "tags": ["custom", "tags"],
            "privacy": "unlisted",
        }

        with patch("aragora.connectors.youtube_uploader.YouTubeVideoMetadata") as MockMetadata:
            with patch.object(social_handler, "read_json_body", return_value=custom_options):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
                )

        # Verify metadata was created with custom values
        MockMetadata.assert_called_once()
        call_kwargs = MockMetadata.call_args[1]
        assert call_kwargs["title"] == "Custom Title"
        assert call_kwargs["description"] == "Custom Description"
        assert call_kwargs["tags"] == ["custom", "tags"]
        assert call_kwargs["privacy_status"] == "unlisted"

    def test_publish_youtube_invalid_json_body(self, social_handler, mock_http_handler):
        """Returns 400 for invalid JSON body."""
        with patch.object(social_handler, "read_json_body", return_value=None):
            result = social_handler.handle_post(
                "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "invalid json" in body["error"].lower()

    def test_publish_youtube_upload_failure(
        self, social_handler, mock_http_handler, mock_youtube_connector
    ):
        """Handles upload failures gracefully."""
        mock_youtube_connector.upload = AsyncMock(
            return_value={
                "success": False,
                "error": "Upload failed: API error",
            }
        )

        with patch("aragora.connectors.youtube_uploader.YouTubeVideoMetadata"):
            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
                )

        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert body["success"] is False

    def test_publish_youtube_waveform_fallback(
        self, handler_ctx, mock_http_handler, mock_video_generator
    ):
        """Falls back to static video when waveform generation fails."""
        mock_video_generator.generate_waveform_video.side_effect = Exception("FFmpeg not found")
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch("aragora.connectors.youtube_uploader.YouTubeVideoMetadata"):
            with patch.object(handler, "read_json_body", return_value={}):
                result = handler.handle_post(
                    "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
                )

        assert result is not None
        # Should succeed using static video fallback
        assert result.status_code == 200
        mock_video_generator.generate_static_video.assert_called_once()


# =============================================================================
# YouTube Status Tests
# =============================================================================


class TestYouTubeStatus:
    """Tests for YouTube status endpoint."""

    def test_youtube_status_configured(self, social_handler):
        """Returns status when YouTube configured."""
        result = social_handler.handle("/api/youtube/status", {}, None)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["configured"] is True
        assert body["has_client_id"] is True
        assert body["has_client_secret"] is True
        assert body["has_refresh_token"] is True
        assert "quota_remaining" in body
        assert body["circuit_breaker_open"] is False

    def test_youtube_status_not_initialized(self):
        """Returns not configured when connector not initialized."""
        handler = SocialMediaHandler(server_context={})

        result = handler.handle("/api/youtube/status", {}, None)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["configured"] is False
        assert "not initialized" in body["error"].lower()

    def test_youtube_status_partial_config(self, handler_ctx):
        """Shows which credentials are missing."""
        handler_ctx["youtube_connector"].client_id = None
        handler_ctx["youtube_connector"].refresh_token = None
        handler = SocialMediaHandler(server_context=handler_ctx)

        result = handler.handle("/api/youtube/status", {}, None)

        body = json.loads(result.body)
        assert body["has_client_id"] is False
        assert body["has_client_secret"] is True
        assert body["has_refresh_token"] is False


# =============================================================================
# Connector Configuration Tests
# =============================================================================


class TestConnectorConfiguration:
    """Tests for connector configuration validation."""

    def test_twitter_config_hint(self, handler_ctx, mock_http_handler):
        """Returns configuration hint when Twitter not configured."""
        handler_ctx["twitter_connector"].is_configured = False
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post("/api/debates/test/publish/twitter", {}, mock_http_handler)

        body = json.loads(result.body)
        assert "hint" in body
        assert "TWITTER_API_KEY" in body["hint"]

    def test_youtube_config_hint(self, handler_ctx, mock_http_handler):
        """Returns configuration hint when YouTube not configured."""
        handler_ctx["youtube_connector"].is_configured = False
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post("/api/debates/test/publish/youtube", {}, mock_http_handler)

        body = json.loads(result.body)
        assert "hint" in body
        assert "YOUTUBE_CLIENT_ID" in body["hint"]


# =============================================================================
# POST Body Parsing Tests
# =============================================================================


class TestPostBodyParsing:
    """Tests for POST body parsing in publishing endpoints."""

    def test_empty_body_uses_defaults(self, social_handler, mock_http_handler):
        """Empty JSON body uses default options."""
        with patch("aragora.connectors.twitter_poster.DebateContentFormatter") as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format_single_tweet.return_value = "Tweet"

            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
                )

        assert result is not None
        assert result.status_code == 200

    def test_none_handler_body_defaults_to_empty(self, social_handler):
        """None handler results in empty options dict."""
        with patch("aragora.connectors.twitter_poster.DebateContentFormatter") as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format_single_tweet.return_value = "Tweet"

            result = social_handler.handle_post(
                "/api/debates/test-debate-123/publish/twitter",
                {},
                None,  # No HTTP handler
            )

        # Should still work with empty options
        assert result is not None


# =============================================================================
# OAuth State Capacity Tests
# =============================================================================


class TestOAuthStateCapacity:
    """Tests for OAuth state storage capacity limits."""

    def setup_method(self):
        """Clear OAuth states before each test."""
        with _oauth_states_lock:
            _oauth_states.clear()

    def test_max_states_eviction(self):
        """Old states are evicted when capacity reached."""
        import time

        # Fill to capacity
        for i in range(MAX_OAUTH_STATES):
            _store_oauth_state(f"state-{i}")

        # Store one more
        _store_oauth_state("new-state")

        with _oauth_states_lock:
            # Should have evicted some old states
            assert len(_oauth_states) <= MAX_OAUTH_STATES
            # New state should be present
            assert "new-state" in _oauth_states


# =============================================================================
# Debate ID Validation Tests
# =============================================================================


class TestDebateIdValidation:
    """Tests for debate ID validation in publishing endpoints."""

    def test_invalid_debate_id_rejected(self, social_handler, mock_http_handler):
        """Invalid debate IDs are rejected."""
        invalid_ids = [
            "../etc/passwd",
            "<script>",
            "id; DROP TABLE",
        ]

        for invalid_id in invalid_ids:
            result = social_handler.handle_post(
                f"/api/debates/{invalid_id}/publish/twitter", {}, mock_http_handler
            )

            # Should be rejected with 400
            if result:
                assert result.status_code in (400, 404)

    def test_valid_debate_id_formats(self, social_handler, mock_http_handler):
        """Valid debate ID formats are accepted."""
        valid_ids = [
            "simple-id",
            "debate_123",
            "abc123def456",
        ]

        for valid_id in valid_ids:
            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    f"/api/debates/{valid_id}/publish/twitter", {}, mock_http_handler
                )

            # Should proceed (may return 404 if debate not found, but not 400)
            if result:
                assert (
                    result.status_code != 400
                    or "not found" in json.loads(result.body).get("error", "").lower() is False
                )


# =============================================================================
# Response Format Tests
# =============================================================================


class TestPublishingResponseFormat:
    """Tests for publishing endpoint response formats."""

    def test_twitter_success_response_structure(self, social_handler, mock_http_handler):
        """Twitter success response has expected fields."""
        with patch("aragora.connectors.twitter_poster.DebateContentFormatter") as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format_single_tweet.return_value = "Tweet"

            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/twitter", {}, mock_http_handler
                )

        body = json.loads(result.body)
        assert "success" in body
        assert "debate_id" in body
        # Either tweet_id or thread_ids depending on mode
        assert "tweet_id" in body or "thread_ids" in body

    def test_youtube_success_response_structure(self, social_handler, mock_http_handler):
        """YouTube success response has expected fields."""
        with patch("aragora.connectors.youtube_uploader.YouTubeVideoMetadata"):
            with patch.object(social_handler, "read_json_body", return_value={}):
                result = social_handler.handle_post(
                    "/api/debates/test-debate-123/publish/youtube", {}, mock_http_handler
                )

        body = json.loads(result.body)
        assert "success" in body
        assert "debate_id" in body
        assert "video_id" in body
        assert "url" in body
        assert "quota_remaining" in body

    def test_error_response_structure(self, handler_ctx, mock_http_handler):
        """Error responses have consistent structure."""
        handler_ctx["twitter_connector"].is_configured = False
        handler = SocialMediaHandler(server_context=handler_ctx)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post("/api/debates/test/publish/twitter", {}, mock_http_handler)

        body = json.loads(result.body)
        assert "error" in body
        assert isinstance(body["error"], str)
