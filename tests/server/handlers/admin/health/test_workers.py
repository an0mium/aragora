"""Tests for background worker and job queue health checks."""

import json
import sys
import types as _types_mod
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

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


class MockHandler:
    """Mock handler for testing health check functions."""

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


class TestWorkerHealthStatus:
    """Tests for worker_health_status function."""

    def test_all_workers_running(self):
        """Test status when all workers are running."""
        from aragora.server.handlers.admin.health.workers import worker_health_status

        handler = MockHandler()

        # Mock gauntlet worker
        mock_gauntlet = MagicMock()
        mock_gauntlet._running = True
        mock_gauntlet._active_jobs = {"job1": MagicMock()}
        mock_gauntlet.max_concurrent = 3
        mock_gauntlet.worker_id = "gauntlet-worker-123"

        # Mock notification dispatcher
        mock_dispatcher = MagicMock()
        mock_dispatcher._worker_task = MagicMock()
        mock_dispatcher._worker_task.done.return_value = False
        mock_dispatcher.config = MagicMock()
        mock_dispatcher.config.queue_enabled = True
        mock_dispatcher.config.max_concurrent_deliveries = 20

        # Mock consensus healing worker
        mock_healing = MagicMock()
        mock_healing._running = True
        mock_healing._candidates_processed = 50
        mock_healing._healed_count = 10

        with (
            patch(
                "aragora.server.startup.workers.get_gauntlet_worker",
                return_value=mock_gauntlet,
            ),
            patch(
                "aragora.control_plane.notifications.get_default_notification_dispatcher",
                return_value=mock_dispatcher,
            ),
            patch(
                "aragora.queue.workers.get_consensus_healing_worker",
                return_value=mock_healing,
            ),
        ):
            result = worker_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"
        assert body["summary"]["running_workers"] == 3
        assert body["summary"]["stopped_workers"] == 0

    def test_some_workers_stopped(self):
        """Test status when some workers are stopped."""
        from aragora.server.handlers.admin.health.workers import worker_health_status

        handler = MockHandler()

        # Mock gauntlet worker running
        mock_gauntlet = MagicMock()
        mock_gauntlet._running = True
        mock_gauntlet._active_jobs = {}
        mock_gauntlet.max_concurrent = 3
        mock_gauntlet.worker_id = "gauntlet-worker-123"

        # Mock notification dispatcher not running
        mock_dispatcher = MagicMock()
        mock_dispatcher._worker_task = None

        with (
            patch(
                "aragora.server.startup.workers.get_gauntlet_worker",
                return_value=mock_gauntlet,
            ),
            patch(
                "aragora.control_plane.notifications.get_default_notification_dispatcher",
                return_value=mock_dispatcher,
            ),
            patch(
                "aragora.queue.workers.get_consensus_healing_worker",
                return_value=None,
            ),
        ):
            result = worker_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "degraded"
        assert body["summary"]["running_workers"] == 1
        assert body["summary"]["stopped_workers"] == 2

    def test_no_workers_running(self):
        """Test status when no workers are running."""
        from aragora.server.handlers.admin.health.workers import worker_health_status

        handler = MockHandler()

        with (
            patch(
                "aragora.server.startup.workers.get_gauntlet_worker",
                return_value=None,
            ),
            patch(
                "aragora.control_plane.notifications.get_default_notification_dispatcher",
                return_value=None,
            ),
            patch(
                "aragora.queue.workers.get_consensus_healing_worker",
                return_value=None,
            ),
        ):
            result = worker_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "unhealthy"
        assert body["summary"]["running_workers"] == 0

    def test_module_import_errors_handled(self):
        """Test that import errors are handled gracefully."""
        from aragora.server.handlers.admin.health.workers import worker_health_status

        handler = MockHandler()

        with (
            patch(
                "aragora.server.startup.workers.get_gauntlet_worker",
                side_effect=ImportError("Module not found"),
            ),
            patch(
                "aragora.control_plane.notifications.get_default_notification_dispatcher",
                side_effect=ImportError("Module not found"),
            ),
            patch(
                "aragora.queue.workers.get_consensus_healing_worker",
                side_effect=ImportError("Module not found"),
            ),
        ):
            result = worker_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["errors"] is not None
        assert len(body["errors"]) >= 1


class TestJobQueueHealthStatus:
    """Tests for job_queue_health_status function."""

    def test_queue_healthy(self):
        """Test status when queue is healthy with low pending count."""
        from aragora.server.handlers.admin.health.workers import job_queue_health_status

        handler = MockHandler()

        mock_store = MagicMock()
        mock_stats = {
            "pending": 5,
            "processing": 2,
            "completed": 100,
            "failed": 0,
            "cancelled": 1,
            "total": 108,
        }

        # Mock the async get_stats method
        async def mock_get_stats():
            return mock_stats

        mock_store.get_stats = mock_get_stats

        with patch(
            "aragora.storage.job_queue_store.get_job_store",
            return_value=mock_store,
        ):
            result = job_queue_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"
        assert body["connected"] is True
        assert body["stats"]["pending"] == 5
        assert body["warnings"] is None

    def test_queue_warning_pending_threshold(self):
        """Test status when pending count exceeds warning threshold."""
        from aragora.server.handlers.admin.health.workers import job_queue_health_status

        handler = MockHandler()

        mock_store = MagicMock()
        mock_stats = {
            "pending": 75,  # Above warning threshold of 50
            "processing": 5,
            "completed": 100,
            "failed": 3,
            "total": 183,
        }

        async def mock_get_stats():
            return mock_stats

        mock_store.get_stats = mock_get_stats

        with patch(
            "aragora.storage.job_queue_store.get_job_store",
            return_value=mock_store,
        ):
            result = job_queue_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "degraded"
        assert body["warnings"] is not None
        assert any("pending jobs" in w.lower() for w in body["warnings"])

    def test_queue_critical_pending_threshold(self):
        """Test status when pending count exceeds critical threshold."""
        from aragora.server.handlers.admin.health.workers import job_queue_health_status

        handler = MockHandler()

        mock_store = MagicMock()
        mock_stats = {
            "pending": 250,  # Above critical threshold of 200
            "processing": 10,
            "completed": 500,
            "failed": 5,
            "total": 765,
        }

        async def mock_get_stats():
            return mock_stats

        mock_store.get_stats = mock_get_stats

        with patch(
            "aragora.storage.job_queue_store.get_job_store",
            return_value=mock_store,
        ):
            result = job_queue_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "critical"
        assert body["warnings"] is not None
        assert any("critical" in w.lower() for w in body["warnings"])

    def test_queue_connection_error(self):
        """Test status when queue connection fails."""
        from aragora.server.handlers.admin.health.workers import job_queue_health_status

        handler = MockHandler()

        with patch(
            "aragora.storage.job_queue_store.get_job_store",
            side_effect=ConnectionError("Redis connection refused"),
        ):
            result = job_queue_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "unhealthy"
        assert body["connected"] is False
        assert body["errors"] is not None

    def test_queue_import_error(self):
        """Test status when job queue module not available."""
        from aragora.server.handlers.admin.health.workers import job_queue_health_status

        handler = MockHandler()

        with patch(
            "aragora.storage.job_queue_store.get_job_store",
            side_effect=ImportError("Module not found"),
        ):
            result = job_queue_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["connected"] is False
        assert body["errors"] is not None


class TestCombinedWorkerQueueHealth:
    """Tests for combined_worker_queue_health function."""

    def test_both_healthy(self):
        """Test combined status when both workers and queue are healthy."""
        from aragora.server.handlers.admin.health.workers import combined_worker_queue_health

        handler = MockHandler()

        # Mock gauntlet worker running
        mock_gauntlet = MagicMock()
        mock_gauntlet._running = True
        mock_gauntlet._active_jobs = {}
        mock_gauntlet.max_concurrent = 3
        mock_gauntlet.worker_id = "gauntlet-worker-123"

        # Mock queue stats
        mock_store = MagicMock()
        mock_stats = {
            "pending": 2,
            "processing": 1,
            "completed": 50,
            "failed": 0,
            "total": 53,
        }

        async def mock_get_stats():
            return mock_stats

        mock_store.get_stats = mock_get_stats

        with (
            patch(
                "aragora.server.startup.workers.get_gauntlet_worker",
                return_value=mock_gauntlet,
            ),
            patch(
                "aragora.control_plane.notifications.get_default_notification_dispatcher",
                return_value=None,
            ),
            patch(
                "aragora.queue.workers.get_consensus_healing_worker",
                return_value=None,
            ),
            patch(
                "aragora.storage.job_queue_store.get_job_store",
                return_value=mock_store,
            ),
        ):
            result = combined_worker_queue_health(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        # At least one worker running = degraded (gauntlet running, others not)
        # Queue healthy, so overall worst is degraded
        assert body["status"] in ["healthy", "degraded"]
        assert "workers" in body
        assert "job_queue" in body

    def test_overall_status_worst_of_both(self):
        """Test that combined status reflects the worst of both."""
        from aragora.server.handlers.admin.health.workers import combined_worker_queue_health

        handler = MockHandler()

        # Mock no workers running (unhealthy)
        # Mock queue with critical pending count

        mock_store = MagicMock()
        mock_stats = {
            "pending": 300,  # Critical
            "processing": 5,
            "completed": 100,
            "failed": 10,
            "total": 415,
        }

        async def mock_get_stats():
            return mock_stats

        mock_store.get_stats = mock_get_stats

        with (
            patch(
                "aragora.server.startup.workers.get_gauntlet_worker",
                return_value=None,
            ),
            patch(
                "aragora.control_plane.notifications.get_default_notification_dispatcher",
                return_value=None,
            ),
            patch(
                "aragora.queue.workers.get_consensus_healing_worker",
                return_value=None,
            ),
            patch(
                "aragora.storage.job_queue_store.get_job_store",
                return_value=mock_store,
            ),
        ):
            result = combined_worker_queue_health(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        # Workers: unhealthy, Queue: critical
        # unhealthy has priority 3, critical has priority 2
        # So overall should be "unhealthy"
        assert body["status"] == "unhealthy"

    def test_response_includes_timestamp(self):
        """Test that response includes ISO timestamp."""
        from aragora.server.handlers.admin.health.workers import combined_worker_queue_health

        handler = MockHandler()

        mock_store = MagicMock()

        async def mock_get_stats():
            return {"pending": 0, "processing": 0, "completed": 0, "failed": 0, "total": 0}

        mock_store.get_stats = mock_get_stats

        with (
            patch(
                "aragora.server.startup.workers.get_gauntlet_worker",
                return_value=None,
            ),
            patch(
                "aragora.control_plane.notifications.get_default_notification_dispatcher",
                return_value=None,
            ),
            patch(
                "aragora.queue.workers.get_consensus_healing_worker",
                return_value=None,
            ),
            patch(
                "aragora.storage.job_queue_store.get_job_store",
                return_value=mock_store,
            ),
        ):
            result = combined_worker_queue_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")


class TestHealthHandlerIntegration:
    """Tests for HealthHandler routing to worker endpoints."""

    def test_worker_routes_registered(self):
        """Test that worker health routes are registered."""
        from aragora.server.handlers.admin.health import HealthHandler

        # Check class-level ROUTES without instantiation
        assert "/api/v1/health/workers" in HealthHandler.ROUTES
        assert "/api/v1/health/job-queue" in HealthHandler.ROUTES
        assert "/api/v1/health/workers/all" in HealthHandler.ROUTES

    def test_can_handle_worker_routes(self):
        """Test that handler can handle worker routes."""
        from aragora.server.handlers.admin.health import HealthHandler

        # Create mock server context
        mock_context = MagicMock()
        handler = HealthHandler(mock_context)

        assert handler.can_handle("/api/v1/health/workers") is True
        assert handler.can_handle("/api/v1/health/job-queue") is True
        assert handler.can_handle("/api/v1/health/workers/all") is True
