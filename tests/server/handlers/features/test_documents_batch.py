"""Tests for Documents Batch Handler."""

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
from unittest.mock import MagicMock, patch

from aragora.server.handlers.features.documents_batch import (
    DocumentsBatchHandler,
    _batch_limiter,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    _batch_limiter._requests.clear()
    yield


@pytest.fixture
def handler():
    """Create handler instance."""
    return DocumentsBatchHandler(ctx={})


class TestDocumentsBatchHandler:
    """Tests for DocumentsBatchHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(DocumentsBatchHandler, "ROUTES")
        routes = DocumentsBatchHandler.ROUTES
        assert "/api/v1/documents/batch" in routes
        assert "/api/v1/documents/batch/status" in routes

    def test_can_handle_batch_routes(self, handler):
        """Test can_handle for batch routes."""
        assert handler.can_handle("/api/v1/documents/batch") is True
        assert handler.can_handle("/api/v1/documents/batch/status") is True

    def test_can_handle_job_routes(self, handler):
        """Test can_handle for job-specific routes."""
        assert handler.can_handle("/api/v1/documents/batch/job123") is True
        assert handler.can_handle("/api/v1/documents/batch/job123/status") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/files/batch") is False
        assert handler.can_handle("/api/v1/invalid/route") is False


class TestBatchUpload:
    """Tests for batch upload endpoint."""

    def test_batch_upload_missing_files(self, handler):
        """Test batch upload requires files."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch.object(handler, "read_json_body", return_value={"workspace_id": "ws1"}),
            patch(
                "aragora.server.handlers.features.documents_batch.require_user_auth",
                lambda f: f,
            ),
        ):
            result = handler._batch_upload(mock_handler)
            assert result.status == 400

    def test_batch_upload_empty_files(self, handler):
        """Test batch upload rejects empty files list."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch.object(
                handler,
                "read_json_body",
                return_value={"workspace_id": "ws1", "files": []},
            ),
            patch(
                "aragora.server.handlers.features.documents_batch.require_user_auth",
                lambda f: f,
            ),
        ):
            result = handler._batch_upload(mock_handler)
            assert result.status == 400


class TestBatchStatus:
    """Tests for batch status endpoint."""

    def test_get_batch_status(self, handler):
        """Test getting batch status."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch("aragora.server.handlers.features.documents_batch.get_batch_job") as mock_get_job,
            patch(
                "aragora.server.handlers.features.documents_batch.require_user_auth",
                lambda f: f,
            ),
        ):
            mock_get_job.return_value = MagicMock(
                job_id="job123",
                status="processing",
                total_files=10,
                processed_files=5,
                to_dict=lambda: {"job_id": "job123", "status": "processing"},
            )

            result = handler.handle("/api/v1/documents/batch/job123", {}, mock_handler)
            assert result.status == 200

    def test_get_batch_status_not_found(self, handler):
        """Test getting status for non-existent batch."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch("aragora.server.handlers.features.documents_batch.get_batch_job") as mock_get_job,
            patch(
                "aragora.server.handlers.features.documents_batch.require_user_auth",
                lambda f: f,
            ),
        ):
            mock_get_job.return_value = None

            result = handler.handle("/api/v1/documents/batch/invalid-job", {}, mock_handler)
            assert result.status == 404


class TestBatchMultipart:
    """Tests for multipart batch upload."""

    def test_multipart_missing_boundary(self, handler):
        """Test multipart requires boundary."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)
        mock_handler.headers = {"Content-Type": "multipart/form-data"}

        with patch(
            "aragora.server.handlers.features.documents_batch.require_user_auth",
            lambda f: f,
        ):
            result = handler._handle_multipart_upload(mock_handler)
            assert result.status == 400


class TestBatchCancel:
    """Tests for canceling batch jobs."""

    def test_cancel_batch_not_found(self, handler):
        """Test canceling non-existent batch."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch("aragora.server.handlers.features.documents_batch.get_batch_job") as mock_get_job,
            patch(
                "aragora.server.handlers.features.documents_batch.require_user_auth",
                lambda f: f,
            ),
        ):
            mock_get_job.return_value = None

            result = handler.handle_delete("/api/v1/documents/batch/invalid-job", {}, mock_handler)
            assert result.status == 404

    def test_cancel_batch_success(self, handler):
        """Test successful batch cancellation."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch("aragora.server.handlers.features.documents_batch.get_batch_job") as mock_get_job,
            patch(
                "aragora.server.handlers.features.documents_batch.cancel_batch_job"
            ) as mock_cancel,
            patch(
                "aragora.server.handlers.features.documents_batch.require_user_auth",
                lambda f: f,
            ),
        ):
            mock_get_job.return_value = MagicMock(job_id="job123", status="processing")
            mock_cancel.return_value = True

            result = handler.handle_delete("/api/v1/documents/batch/job123", {}, mock_handler)
            assert result.status == 200


class TestBatchRateLimiting:
    """Tests for batch rate limiting."""

    def test_rate_limiter_exists(self):
        """Test that rate limiter is configured."""
        assert _batch_limiter is not None
        assert _batch_limiter.requests_per_minute == 20

    def test_rate_limit_exceeded(self, handler):
        """Test rate limit enforcement."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Exhaust rate limit
        for _ in range(21):
            _batch_limiter.is_allowed("127.0.0.1")

        with patch(
            "aragora.server.handlers.features.documents_batch.get_client_ip",
            return_value="127.0.0.1",
        ):
            result = handler.handle("/api/v1/documents/batch/status", {}, mock_handler)
            assert result.status == 429
