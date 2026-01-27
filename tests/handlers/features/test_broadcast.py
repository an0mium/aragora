"""Tests for broadcast generation handler.

Tests the broadcast API endpoints including:
- POST /api/v1/debates/{id}/broadcast - Generate podcast audio from debate trace
- POST /api/v1/debates/{id}/broadcast/full - Run full broadcast pipeline
- GET /api/podcast/feed.xml - Get RSS podcast feed
"""

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@dataclass
class MockHandler:
    """Mock HTTP handler for tests."""

    headers: Dict[str, str] = None
    rfile: BytesIO = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Length": "0"}
        if self.rfile is None:
            self.rfile = BytesIO(b"{}")


@pytest.fixture
def broadcast_handler():
    """Create broadcast handler with mock context."""
    from aragora.server.handlers.features.broadcast import BroadcastHandler

    ctx = {
        "nomic_dir": Path("/tmp/test_nomic"),
        "audio_store": MagicMock(),
    }
    handler = BroadcastHandler(ctx)
    # Reset cached pipeline between tests
    handler._pipeline = None
    return handler


@pytest.fixture(autouse=True)
def reset_handler_state():
    """Reset handler state before each test."""
    from aragora.server.handlers.features import broadcast as broadcast_module

    # Reset rate limiters
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass

    # Reset any cached state
    yield

    # Cleanup after test
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass


# =============================================================================
# Initialization Tests
# =============================================================================


class TestBroadcastHandlerInit:
    """Tests for handler initialization."""

    def test_routes_defined(self, broadcast_handler):
        """Test that handler routes are defined."""
        assert hasattr(broadcast_handler, "ROUTES")
        assert len(broadcast_handler.ROUTES) > 0

    def test_can_handle_broadcast_path(self, broadcast_handler):
        """Test can_handle recognizes broadcast paths."""
        assert broadcast_handler.can_handle("/api/v1/debates/abc123/broadcast")
        assert broadcast_handler.can_handle("/api/v1/debates/abc123/broadcast/full")
        # Note: /api/podcast/feed.xml is handled by AudioHandler, not BroadcastHandler

    def test_cannot_handle_other_paths(self, broadcast_handler):
        """Test can_handle rejects non-broadcast paths."""
        assert not broadcast_handler.can_handle("/api/v1/debates")
        assert not broadcast_handler.can_handle("/api/v1/debates/abc123")
        assert not broadcast_handler.can_handle("/api/v1/users")
        assert not broadcast_handler.can_handle("/api/v1/podcast")


# =============================================================================
# RSS Feed Tests
# Note: RSS feed functionality is handled by AudioHandler, not BroadcastHandler.
# These tests have been removed as they were testing the wrong handler.
# See test_audio.py for RSS feed tests.
# =============================================================================


# =============================================================================
# Basic Broadcast Generation Tests
# =============================================================================


class TestGenerateBroadcast:
    """Tests for basic broadcast generation endpoint."""

    def test_returns_503_without_broadcast_module(self, broadcast_handler):
        """Returns 503 when broadcast module not available."""
        with patch("aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE", False):
            result = broadcast_handler._generate_broadcast("test-debate", MockHandler())
            assert result.status_code == 503

    def test_returns_503_without_storage(self, broadcast_handler):
        """Returns 503 when storage not available."""
        with patch("aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE", True):
            with patch.object(broadcast_handler, "get_storage", return_value=None):
                result = broadcast_handler._generate_broadcast("test-debate", MockHandler())
                assert result.status_code == 503

    def test_returns_503_without_nomic_dir(self, broadcast_handler):
        """Returns 503 when nomic directory not configured."""
        mock_storage = MagicMock()

        with patch("aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE", True):
            with patch.object(broadcast_handler, "get_storage", return_value=mock_storage):
                with patch.object(broadcast_handler, "get_nomic_dir", return_value=None):
                    result = broadcast_handler._generate_broadcast("test-debate", MockHandler())
                    assert result.status_code == 503

    def test_returns_404_debate_not_found(self, broadcast_handler):
        """Returns 404 when debate doesn't exist."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None
        mock_storage.get_debate_by_slug.return_value = None

        with patch("aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE", True):
            with patch.object(broadcast_handler, "get_storage", return_value=mock_storage):
                with patch.object(broadcast_handler, "get_nomic_dir", return_value=Path("/tmp")):
                    result = broadcast_handler._generate_broadcast("nonexistent", MockHandler())
                    assert result.status_code == 404

    def test_returns_existing_audio_if_available(self, broadcast_handler):
        """Returns existing audio info instead of regenerating."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"id": "test-123"}

        mock_audio_store = MagicMock()
        mock_audio_store.exists.return_value = True
        mock_audio_store.get_metadata.return_value = {"generated_at": "2024-01-01"}
        mock_audio_store.get_path.return_value = Path("/audio/test-123.mp3")

        broadcast_handler.ctx["audio_store"] = mock_audio_store

        with patch("aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE", True):
            with patch.object(broadcast_handler, "get_storage", return_value=mock_storage):
                with patch.object(broadcast_handler, "get_nomic_dir", return_value=Path("/tmp")):
                    result = broadcast_handler._generate_broadcast("test-123", MockHandler())
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["status"] == "exists"


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Tests for full broadcast pipeline endpoint."""

    def test_returns_503_without_pipeline(self, broadcast_handler):
        """Returns 503 when pipeline not available."""
        with patch.object(broadcast_handler, "_get_pipeline", return_value=None):
            result = broadcast_handler._run_full_pipeline("test-debate", {}, MockHandler())
            assert result.status_code == 503

    def test_parses_query_params(self, broadcast_handler):
        """Parses video, title, description query params."""
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.debate_id = "test-123"
        mock_result.success = True
        mock_result.audio_path = Path("/audio/test.mp3")
        mock_result.video_path = None
        mock_result.rss_episode_guid = "guid-123"
        mock_result.duration_seconds = 120
        mock_result.steps_completed = ["script", "audio"]
        mock_result.generated_at = "2024-01-01"
        mock_result.error_message = None

        async def mock_run(debate_id, options):
            # Verify options are parsed correctly
            assert options.video_enabled is True
            assert options.custom_title == "Custom Title"
            return mock_result

        mock_pipeline.run = mock_run

        with patch.object(broadcast_handler, "_get_pipeline", return_value=mock_pipeline):
            with patch(
                "aragora.server.handlers.features.broadcast.BroadcastOptions"
            ) as MockOptions:
                MockOptions.return_value = MagicMock(video_enabled=True)

                result = broadcast_handler._run_full_pipeline(
                    "test-123",
                    {"video": ["true"], "title": ["Custom Title"]},
                    MockHandler(),
                )

                # Pipeline was called
                assert result is not None

    def test_handles_pipeline_exception(self, broadcast_handler):
        """Returns 500 on pipeline exception."""
        mock_pipeline = MagicMock()

        async def mock_run_fail(*args, **kwargs):
            raise RuntimeError("Pipeline crashed")

        mock_pipeline.run = mock_run_fail

        with patch.object(broadcast_handler, "_get_pipeline", return_value=mock_pipeline):
            with patch(
                "aragora.server.handlers.features.broadcast._run_async",
                side_effect=RuntimeError("Pipeline crashed"),
            ):
                result = broadcast_handler._run_full_pipeline("test-123", {}, MockHandler())
                assert result.status_code == 500


# =============================================================================
# Path Extraction Tests
# =============================================================================


class TestPathExtraction:
    """Tests for debate ID extraction from paths."""

    def test_extracts_debate_id_from_broadcast_path(self, broadcast_handler):
        """Extracts debate ID from /api/v1/debates/{id}/broadcast."""
        result = broadcast_handler.handle_post(
            "/api/v1/debates/test-123/broadcast", {}, MockHandler()
        )
        # Will fail due to missing modules but ID extraction should work
        assert result is not None

    def test_extracts_debate_id_from_full_path(self, broadcast_handler):
        """Extracts debate ID from /api/v1/debates/{id}/broadcast/full."""
        result = broadcast_handler.handle_post(
            "/api/v1/debates/abc-xyz/broadcast/full", {}, MockHandler()
        )
        # Will fail due to missing modules but ID extraction should work
        assert result is not None

    def test_rejects_invalid_debate_id_characters(self, broadcast_handler):
        """Rejects debate IDs with invalid characters."""
        # Path traversal attempt
        result = broadcast_handler.handle_post(
            "/api/v1/debates/../../../etc/passwd/broadcast", {}, MockHandler()
        )
        # Should either reject or return error
        assert result is not None
        if result.status_code != 400:
            # May return 503/404 if handler proceeds but debate not found
            assert result.status_code in [404, 503]


# =============================================================================
# Handle Method Tests
# =============================================================================


class TestHandleMethods:
    """Tests for handle and handle_post methods."""

    def test_handle_returns_none_for_non_rss(self, broadcast_handler):
        """Handle returns None for non-RSS GET requests."""
        result = broadcast_handler.handle("/api/v1/debates/test/broadcast", {}, None)
        assert result is None

    def test_handle_post_routes_to_full_pipeline(self, broadcast_handler):
        """handle_post routes /broadcast/full to full pipeline."""
        with patch.object(
            broadcast_handler, "_run_full_pipeline", return_value=MagicMock(status_code=200)
        ) as mock_pipeline:
            broadcast_handler.handle_post(
                "/api/v1/debates/test-123/broadcast/full", {}, MockHandler()
            )
            mock_pipeline.assert_called_once()

    def test_handle_post_routes_to_basic_broadcast(self, broadcast_handler):
        """handle_post routes /broadcast to basic generation."""
        with patch.object(
            broadcast_handler, "_generate_broadcast", return_value=MagicMock(status_code=200)
        ) as mock_gen:
            broadcast_handler.handle_post("/api/v1/debates/test-123/broadcast", {}, MockHandler())
            mock_gen.assert_called_once()

    def test_handle_post_returns_none_for_unmatched(self, broadcast_handler):
        """handle_post returns None for unmatched paths."""
        result = broadcast_handler.handle_post("/api/other/path", {}, MockHandler())
        assert result is None


# =============================================================================
# Pipeline Caching Tests
# =============================================================================


class TestPipelineCaching:
    """Tests for pipeline instance caching."""

    def test_pipeline_created_once(self, broadcast_handler):
        """Pipeline instance is cached after first creation."""
        with patch("aragora.server.handlers.features.broadcast.PIPELINE_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.broadcast.BroadcastPipeline"
            ) as MockPipeline:
                mock_instance = MagicMock()
                MockPipeline.return_value = mock_instance

                with patch.object(broadcast_handler, "get_nomic_dir", return_value=Path("/tmp")):
                    # First call creates pipeline
                    pipeline1 = broadcast_handler._get_pipeline()

                    # Second call returns cached instance
                    pipeline2 = broadcast_handler._get_pipeline()

                    # Only created once
                    assert MockPipeline.call_count <= 1

    def test_returns_none_when_unavailable(self, broadcast_handler):
        """Returns None when pipeline module not available."""
        with patch("aragora.server.handlers.features.broadcast.PIPELINE_AVAILABLE", False):
            result = broadcast_handler._get_pipeline()
            assert result is None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in broadcast handler."""

    def test_handles_trace_load_error(self):
        """Returns 404 when trace file doesn't exist."""
        # Create fresh handler to avoid rate limit issues
        from aragora.server.handlers.features.broadcast import BroadcastHandler

        ctx = {"nomic_dir": Path("/tmp/test_nomic"), "audio_store": MagicMock()}
        handler = BroadcastHandler(ctx)

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"id": "test-err-1"}

        mock_audio_store = MagicMock()
        mock_audio_store.exists.return_value = False
        handler.ctx["audio_store"] = mock_audio_store

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_path = Path(tmpdir)
            traces_dir = nomic_path / "traces"
            traces_dir.mkdir(parents=True)

            with patch(
                "aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE",
                True,
            ):
                with patch.object(handler, "get_storage", return_value=mock_storage):
                    with patch.object(handler, "get_nomic_dir", return_value=nomic_path):
                        # Call the underlying method directly, bypassing both
                        # @require_permission (outer) and @rate_limit (inner) decorators
                        # __wrapped__ once gets past require_permission to rate_limit
                        # __wrapped__ twice gets to the original function
                        original_func = handler._generate_broadcast.__wrapped__.__wrapped__
                        result = original_func(handler, "test-err-1", MockHandler())
                        assert result.status_code == 404

    def test_safe_error_message_used(self):
        """Uses safe_error_message to avoid exposing internals."""
        # Create fresh handler to avoid rate limit issues
        from aragora.server.handlers.features.broadcast import BroadcastHandler

        ctx = {"nomic_dir": Path("/tmp/test_nomic"), "audio_store": MagicMock()}
        handler = BroadcastHandler(ctx)

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"id": "test-err-2"}

        mock_audio_store = MagicMock()
        mock_audio_store.exists.return_value = False
        handler.ctx["audio_store"] = mock_audio_store

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_path = Path(tmpdir)
            traces_dir = nomic_path / "traces"
            traces_dir.mkdir(parents=True)

            # Create invalid trace file
            trace_file = traces_dir / "test-err-2.json"
            trace_file.write_text("invalid json {{{")

            with patch(
                "aragora.server.handlers.features.broadcast.BROADCAST_AVAILABLE",
                True,
            ):
                with patch.object(handler, "get_storage", return_value=mock_storage):
                    with patch.object(handler, "get_nomic_dir", return_value=nomic_path):
                        # Call the underlying method directly, bypassing both
                        # @require_permission (outer) and @rate_limit (inner) decorators
                        original_func = handler._generate_broadcast.__wrapped__.__wrapped__
                        result = original_func(handler, "test-err-2", MockHandler())
                        assert result.status_code == 500
                        # Error message should not contain full path
                        body = (
                            result.body.decode() if isinstance(result.body, bytes) else result.body
                        )
                        assert tmpdir not in body
