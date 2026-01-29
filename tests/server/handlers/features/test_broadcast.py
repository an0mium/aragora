"""Tests for Broadcast Handler."""

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

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.broadcast import (
    BroadcastHandler,
    _broadcast_limiter,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    _broadcast_limiter._buckets.clear()
    yield


@pytest.fixture
def handler():
    """Create handler instance."""
    return BroadcastHandler({})


class TestBroadcastHandler:
    """Tests for BroadcastHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(BroadcastHandler, "ROUTES")
        routes = BroadcastHandler.ROUTES
        assert "/api/v1/broadcast/generate" in routes
        assert "/api/v1/broadcast/jobs" in routes
        assert "/api/v1/broadcast/voices" in routes

    def test_can_handle_broadcast_routes(self, handler):
        """Test can_handle for broadcast routes."""
        assert handler.can_handle("/api/v1/broadcast/generate") is True
        assert handler.can_handle("/api/v1/broadcast/jobs") is True
        assert handler.can_handle("/api/v1/broadcast/voices") is True

    def test_can_handle_job_routes(self, handler):
        """Test can_handle for job-specific routes."""
        assert handler.can_handle("/api/v1/broadcast/jobs/job123") is True
        assert handler.can_handle("/api/v1/broadcast/jobs/job123/cancel") is True
        assert handler.can_handle("/api/v1/broadcast/jobs/job123/download") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/podcast/") is False
        assert handler.can_handle("/api/v1/invalid/route") is False


class TestBroadcastVoices:
    """Tests for voices endpoint."""

    def test_get_voices(self, handler):
        """Test listing available voices."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/broadcast/voices", {}, mock_handler)
        assert result.status_code == 200

        import json

        body = json.loads(result.body)
        assert "voices" in body
        assert len(body["voices"]) > 0


class TestBroadcastGenerate:
    """Tests for broadcast generation."""

    def test_generate_missing_debate_id(self, handler):
        """Test generate requires debate_id."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch.object(handler, "read_json_body", return_value={}),
            patch(
                "aragora.server.handlers.features.broadcast.require_user_auth",
                lambda f: f,
            ),
        ):
            result = handler.handle_post("/api/v1/broadcast/generate", {}, mock_handler)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_generate_debate_not_found(self, handler):
        """Test generate fails when debate not found."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch.object(handler, "read_json_body", return_value={"debate_id": "invalid-debate"}),
            patch(
                "aragora.server.handlers.features.broadcast.require_user_auth",
                lambda f: f,
            ),
            patch("aragora.server.handlers.features.broadcast.get_debate") as mock_get_debate,
        ):
            mock_get_debate.return_value = None

            result = handler.handle_post("/api/v1/broadcast/generate", {}, mock_handler)
            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_generate_success(self, handler):
        """Test successful broadcast generation."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        mock_debate = MagicMock()
        mock_debate.id = "debate123"
        mock_debate.topic = "Test Topic"
        mock_debate.rounds = []

        with (
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "debate_id": "debate123",
                    "style": "podcast",
                    "voices": ["alloy", "echo"],
                },
            ),
            patch(
                "aragora.server.handlers.features.broadcast.require_user_auth",
                lambda f: f,
            ),
            patch("aragora.server.handlers.features.broadcast.get_debate") as mock_get_debate,
            patch(
                "aragora.server.handlers.features.broadcast.generate_broadcast",
                new_callable=AsyncMock,
            ) as mock_generate,
        ):
            mock_get_debate.return_value = mock_debate
            mock_generate.return_value = MagicMock(job_id="job123", status="processing")

            result = handler.handle_post("/api/v1/broadcast/generate", {}, mock_handler)
            assert result.status_code == 202  # Accepted


class TestBroadcastJobs:
    """Tests for broadcast jobs."""

    def test_list_jobs(self, handler):
        """Test listing broadcast jobs."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch(
            "aragora.server.handlers.features.broadcast.get_broadcast_jobs"
        ) as mock_get_jobs:
            mock_get_jobs.return_value = []

            result = handler.handle("/api/v1/broadcast/jobs", {}, mock_handler)
            assert result.status_code == 200

    def test_get_job_status(self, handler):
        """Test getting job status."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.broadcast.get_broadcast_job") as mock_get_job:
            mock_get_job.return_value = MagicMock(
                job_id="job123",
                status="completed",
                to_dict=lambda: {"job_id": "job123", "status": "completed"},
            )

            result = handler.handle("/api/v1/broadcast/jobs/job123", {}, mock_handler)
            assert result.status_code == 200

    def test_get_job_not_found(self, handler):
        """Test getting non-existent job."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.broadcast.get_broadcast_job") as mock_get_job:
            mock_get_job.return_value = None

            result = handler.handle("/api/v1/broadcast/jobs/invalid-job", {}, mock_handler)
            assert result.status_code == 404


class TestBroadcastCancel:
    """Tests for canceling broadcast jobs."""

    def test_cancel_job_not_found(self, handler):
        """Test canceling non-existent job."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.broadcast.get_broadcast_job") as mock_get_job:
            mock_get_job.return_value = None

            result = handler.handle_post(
                "/api/v1/broadcast/jobs/invalid-job/cancel", {}, mock_handler
            )
            assert result.status_code == 404

    def test_cancel_job_success(self, handler):
        """Test successful job cancellation."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch("aragora.server.handlers.features.broadcast.get_broadcast_job") as mock_get_job,
            patch("aragora.server.handlers.features.broadcast.cancel_broadcast_job") as mock_cancel,
        ):
            mock_get_job.return_value = MagicMock(job_id="job123", status="processing")
            mock_cancel.return_value = True

            result = handler.handle_post("/api/v1/broadcast/jobs/job123/cancel", {}, mock_handler)
            assert result.status_code == 200


class TestBroadcastDownload:
    """Tests for downloading broadcast audio."""

    def test_download_job_not_found(self, handler):
        """Test download non-existent job."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.broadcast.get_broadcast_job") as mock_get_job:
            mock_get_job.return_value = None

            result = handler.handle("/api/v1/broadcast/jobs/invalid-job/download", {}, mock_handler)
            assert result.status_code == 404

    def test_download_job_not_completed(self, handler):
        """Test download job that isn't completed."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.broadcast.get_broadcast_job") as mock_get_job:
            mock_get_job.return_value = MagicMock(
                job_id="job123", status="processing", audio_path=None
            )

            result = handler.handle("/api/v1/broadcast/jobs/job123/download", {}, mock_handler)
            assert result.status_code == 400


class TestBroadcastRateLimiting:
    """Tests for broadcast rate limiting."""

    def test_rate_limiter_exists(self):
        """Test that rate limiter is configured."""
        assert _broadcast_limiter is not None
        assert _broadcast_limiter.requests_per_minute == 10

    def test_rate_limit_exceeded(self, handler):
        """Test rate limit enforcement."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Exhaust rate limit
        for _ in range(11):
            _broadcast_limiter.is_allowed("127.0.0.1")

        with patch(
            "aragora.server.handlers.features.broadcast.get_client_ip",
            return_value="127.0.0.1",
        ):
            result = handler.handle("/api/v1/broadcast/voices", {}, mock_handler)
            assert result.status_code == 429
