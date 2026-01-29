"""Tests for Broadcast Handler.

Tests cover handler creation, route definitions, can_handle, and the POST endpoints
for broadcast generation.
"""

import sys
import types as _types_mod
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

from aragora.server.handlers.features.broadcast import (
    BROADCAST_AVAILABLE,
    BroadcastHandler,
    PIPELINE_AVAILABLE,
)


@pytest.fixture
def handler():
    """Create handler instance with minimal context."""
    return BroadcastHandler(server_context={})


@pytest.fixture
def handler_with_storage():
    """Create handler with mocked storage and nomic_dir."""
    mock_storage = MagicMock()
    mock_nomic_dir = Path("/tmp/test_nomic")
    mock_audio_store = MagicMock()

    ctx = {
        "storage": mock_storage,
        "nomic_dir": mock_nomic_dir,
        "audio_store": mock_audio_store,
    }
    return BroadcastHandler(server_context=ctx)


class TestBroadcastHandler:
    """Tests for BroadcastHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None
        assert isinstance(handler, BroadcastHandler)

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(BroadcastHandler, "ROUTES")
        routes = BroadcastHandler.ROUTES
        # Current handler uses debates/*/broadcast pattern
        assert "/api/v1/debates/*/broadcast" in routes
        assert "/api/v1/debates/*/broadcast/full" in routes

    def test_can_handle_broadcast_routes(self, handler):
        """Test can_handle for debate broadcast routes."""
        # Valid broadcast routes
        assert handler.can_handle("/api/v1/debates/debate123/broadcast") is True
        assert handler.can_handle("/api/v1/debates/my-debate/broadcast") is True
        assert handler.can_handle("/api/v1/debates/test_id/broadcast") is True

    def test_can_handle_full_pipeline_routes(self, handler):
        """Test can_handle for full pipeline routes."""
        assert handler.can_handle("/api/v1/debates/debate123/broadcast/full") is True
        assert handler.can_handle("/api/v1/debates/my-debate/broadcast/full") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        # Old API routes should not be handled
        assert handler.can_handle("/api/v1/broadcast/generate") is False
        assert handler.can_handle("/api/v1/broadcast/jobs") is False
        assert handler.can_handle("/api/v1/broadcast/voices") is False

        # Other invalid routes
        assert handler.can_handle("/api/v1/podcast/") is False
        assert handler.can_handle("/api/v1/debates/debate123") is False
        assert handler.can_handle("/api/v1/invalid/route") is False

    def test_handle_get_returns_none(self, handler):
        """Test that GET requests return None (POST only handler)."""
        mock_handler = MagicMock()
        result = handler.handle("/api/v1/debates/debate123/broadcast", {}, mock_handler)
        assert result is None


class TestBroadcastAvailability:
    """Tests for broadcast module availability flags."""

    def test_broadcast_available_is_bool(self):
        """Test that BROADCAST_AVAILABLE is a boolean."""
        assert isinstance(BROADCAST_AVAILABLE, bool)

    def test_pipeline_available_is_bool(self):
        """Test that PIPELINE_AVAILABLE is a boolean."""
        assert isinstance(PIPELINE_AVAILABLE, bool)


class TestBroadcastGenerate:
    """Tests for POST /api/v1/debates/{debate_id}/broadcast endpoint."""

    def test_generate_broadcast_not_available(self, handler):
        """Test response when broadcast module is not available."""
        mock_http_handler = MagicMock()
        mock_http_handler.headers = {}

        with patch.object(handler, "get_storage", return_value=MagicMock()):
            with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp")):
                with patch(
                    "aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE",
                    False,
                ):
                    result = handler.handle_post(
                        "/api/v1/debates/debate123/broadcast", {}, mock_http_handler
                    )
                    assert result is not None
                    assert result.status_code == 503
                    assert b"not available" in result.body.lower()

    def test_generate_broadcast_no_storage(self, handler):
        """Test response when storage is not configured."""
        mock_http_handler = MagicMock()
        mock_http_handler.headers = {}

        with patch.object(handler, "get_storage", return_value=None):
            with patch(
                "aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE",
                True,
            ):
                result = handler.handle_post(
                    "/api/v1/debates/debate123/broadcast", {}, mock_http_handler
                )
                assert result is not None
                assert result.status_code == 503
                assert b"storage" in result.body.lower()

    def test_generate_broadcast_no_nomic_dir(self, handler):
        """Test response when nomic directory is not configured."""
        mock_http_handler = MagicMock()
        mock_http_handler.headers = {}
        mock_storage = MagicMock()

        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                with patch(
                    "aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE",
                    True,
                ):
                    result = handler.handle_post(
                        "/api/v1/debates/debate123/broadcast", {}, mock_http_handler
                    )
                    assert result is not None
                    assert result.status_code == 503
                    assert b"nomic" in result.body.lower()

    def test_generate_broadcast_debate_not_found(self, handler_with_storage):
        """Test response when debate is not found."""
        mock_http_handler = MagicMock()
        mock_http_handler.headers = {}

        storage = handler_with_storage.ctx["storage"]
        storage.get_debate.return_value = None
        storage.get_debate_by_slug.return_value = None

        with patch(
            "aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE",
            True,
        ):
            result = handler_with_storage.handle_post(
                "/api/v1/debates/nonexistent/broadcast", {}, mock_http_handler
            )
            assert result is not None
            assert result.status_code == 404
            assert b"not found" in result.body.lower()

    def test_generate_broadcast_existing_audio(self, handler_with_storage):
        """Test response when audio already exists."""
        mock_http_handler = MagicMock()
        mock_http_handler.headers = {}

        storage = handler_with_storage.ctx["storage"]
        storage.get_debate.return_value = {"id": "debate123", "topic": "Test"}

        audio_store = handler_with_storage.ctx["audio_store"]
        audio_store.exists.return_value = True
        audio_store.get_metadata.return_value = {"generated_at": "2024-01-01T00:00:00"}
        audio_store.get_path.return_value = Path("/tmp/audio/debate123.mp3")

        with patch(
            "aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE",
            True,
        ):
            result = handler_with_storage.handle_post(
                "/api/v1/debates/debate123/broadcast", {}, mock_http_handler
            )
            assert result is not None
            assert result.status_code == 200

            import json

            body = json.loads(result.body)
            assert body["status"] == "exists"
            assert body["debate_id"] == "debate123"
            assert "/audio/debate123.mp3" in body["audio_url"]


class TestBroadcastFullPipeline:
    """Tests for POST /api/v1/debates/{debate_id}/broadcast/full endpoint."""

    def test_full_pipeline_not_available(self, handler):
        """Test response when pipeline is not available."""
        mock_http_handler = MagicMock()
        mock_http_handler.headers = {}

        with patch.object(handler, "_get_pipeline", return_value=None):
            result = handler.handle_post(
                "/api/v1/debates/debate123/broadcast/full", {}, mock_http_handler
            )
            assert result is not None
            assert result.status_code == 503
            assert b"not available" in result.body.lower()

    def test_full_pipeline_with_options(self, handler):
        """Test full pipeline with query params."""
        mock_http_handler = MagicMock()
        mock_http_handler.headers = {}

        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.debate_id = "debate123"
        mock_result.success = True
        mock_result.audio_path = Path("/tmp/debate123.mp3")
        mock_result.video_path = None
        mock_result.rss_episode_guid = "guid-123"
        mock_result.duration_seconds = 120
        mock_result.steps_completed = ["audio"]
        mock_result.generated_at = "2024-01-01T00:00:00"
        mock_result.error_message = None

        mock_pipeline.run = AsyncMock(return_value=mock_result)

        query_params = {
            "video": "false",
            "rss": "true",
            "title": "Custom Title",
        }

        with patch.object(handler, "_get_pipeline", return_value=mock_pipeline):
            with patch(
                "aragora.server.handlers.features.broadcast._run_async",
                side_effect=lambda coro: mock_result,
            ):
                result = handler.handle_post(
                    "/api/v1/debates/debate123/broadcast/full",
                    query_params,
                    mock_http_handler,
                )
                assert result is not None
                assert result.status_code == 200

                import json

                body = json.loads(result.body)
                assert body["debate_id"] == "debate123"
                assert body["success"] is True


class TestBroadcastPathExtraction:
    """Tests for debate_id path parameter extraction."""

    def test_extract_debate_id_basic(self, handler):
        """Test extracting debate_id from broadcast path."""
        # Path: /api/v1/debates/{debate_id}/broadcast
        # Parts: ['', 'api', 'v1', 'debates', 'debate123', 'broadcast']
        # Index:    0     1      2       3          4            5
        debate_id, err = handler.extract_path_param(
            "/api/v1/debates/debate123/broadcast", 4, "debate_id"
        )
        assert err is None
        assert debate_id == "debate123"

    def test_extract_debate_id_with_slug(self, handler):
        """Test extracting slug-style debate_id."""
        debate_id, err = handler.extract_path_param(
            "/api/v1/debates/my-awesome-debate/broadcast", 4, "debate_id"
        )
        assert err is None
        assert debate_id == "my-awesome-debate"

    def test_extract_debate_id_from_full_path(self, handler):
        """Test extracting debate_id from full pipeline path."""
        debate_id, err = handler.extract_path_param(
            "/api/v1/debates/debate456/broadcast/full", 4, "debate_id"
        )
        assert err is None
        assert debate_id == "debate456"


class TestBroadcastHandlerMethods:
    """Tests for BroadcastHandler utility methods."""

    def test_get_pipeline_returns_none_when_unavailable(self, handler):
        """Test _get_pipeline returns None when pipeline module unavailable."""
        with patch(
            "aragora.server.handlers.features.broadcast.PIPELINE_AVAILABLE",
            False,
        ):
            result = handler._get_pipeline()
            assert result is None

    def test_get_pipeline_creates_instance(self, handler):
        """Test _get_pipeline creates pipeline when available."""
        mock_nomic_dir = Path("/tmp/test_nomic")
        handler.ctx["audio_store"] = MagicMock()

        with patch(
            "aragora.server.handlers.features.broadcast.PIPELINE_AVAILABLE",
            True,
        ):
            with patch.object(handler, "get_nomic_dir", return_value=mock_nomic_dir):
                with patch(
                    "aragora.server.handlers.features.broadcast.BroadcastPipeline"
                ) as mock_class:
                    mock_instance = MagicMock()
                    mock_class.return_value = mock_instance

                    result = handler._get_pipeline()
                    assert result == mock_instance
                    mock_class.assert_called_once()

    def test_get_pipeline_returns_cached_instance(self, handler):
        """Test _get_pipeline returns cached instance on subsequent calls."""
        mock_pipeline = MagicMock()
        handler._pipeline = mock_pipeline

        with patch(
            "aragora.server.handlers.features.broadcast.PIPELINE_AVAILABLE",
            True,
        ):
            result = handler._get_pipeline()
            assert result == mock_pipeline
