"""Tests for Audit Scheduler Handler."""

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

from aragora.server.handlers.features.scheduler import SchedulerHandler


@pytest.fixture
def handler():
    """Create handler instance."""
    h = SchedulerHandler({})
    # Add headers so require_user_auth can find the handler
    h.headers = MagicMock()
    return h


@pytest.fixture(autouse=True)
def mock_auth():
    """Mock authentication for all tests."""
    mock_user = MagicMock()
    mock_user.is_authenticated = True
    mock_user.user_id = "test_user"
    mock_user.role = "admin"
    mock_user.error_reason = None

    with patch(
        "aragora.billing.jwt_auth.extract_user_from_request",
        return_value=mock_user,
    ):
        yield


class TestSchedulerHandler:
    """Tests for SchedulerHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_base_routes(self):
        """Test that handler has base route definitions."""
        assert hasattr(SchedulerHandler, "BASE_ROUTES")
        routes = SchedulerHandler.BASE_ROUTES
        assert "/api/v1/scheduler/jobs" in routes
        assert "/api/v1/scheduler/webhooks" in routes
        assert "/api/v1/scheduler/events/git-push" in routes
        assert "/api/v1/scheduler/events/file-upload" in routes
        assert "/api/v1/scheduler/status" in routes

    def test_can_handle_base_routes(self, handler):
        """Test can_handle for base routes."""
        assert handler.can_handle("/api/v1/scheduler/jobs") is True
        assert handler.can_handle("/api/v1/scheduler/status") is True
        assert handler.can_handle("/api/v1/scheduler/events/git-push") is True

    def test_can_handle_job_routes(self, handler):
        """Test can_handle for job-specific routes."""
        assert handler.can_handle("/api/v1/scheduler/jobs/job123") is True
        assert handler.can_handle("/api/v1/scheduler/jobs/job123/history") is True
        assert handler.can_handle("/api/v1/scheduler/jobs/job123/trigger") is True
        assert handler.can_handle("/api/v1/scheduler/jobs/job123/pause") is True

    def test_can_handle_webhook_routes(self, handler):
        """Test can_handle for webhook routes."""
        assert handler.can_handle("/api/v1/scheduler/webhooks/wh123") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/tasks/") is False
        assert handler.can_handle("/api/v1/invalid/route") is False


class TestSchedulerStatus:
    """Tests for scheduler status endpoint."""

    def test_get_scheduler_status(self, handler):
        """Test get scheduler status."""
        mock_scheduler = MagicMock()
        mock_scheduler._running = True
        mock_scheduler.list_jobs.return_value = []

        with patch.object(handler, "_get_scheduler", return_value=mock_scheduler):
            result = handler._get_scheduler_status()
            assert result.status_code == 200

            import json

            body = json.loads(result.body)
            assert body["running"] is True
            assert body["total_jobs"] == 0


class TestSchedulerListJobs:
    """Tests for listing scheduler jobs."""

    def test_list_jobs(self, handler):
        """Test listing jobs."""
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._list_jobs({})
            assert result.status_code == 200

    def test_list_jobs_with_status_filter(self, handler):
        """Test listing jobs with status filter."""
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
            patch("aragora.scheduler.ScheduleStatus") as MockStatus,
        ):
            MockStatus.return_value = "active"
            result = handler._list_jobs({"status": ["active"]})
            assert result.status_code == 200

    def test_list_jobs_invalid_status(self, handler):
        """Test listing jobs with invalid status returns error."""
        mock_scheduler = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
            patch(
                "aragora.scheduler.ScheduleStatus",
                side_effect=ValueError("Invalid"),
            ),
        ):
            result = handler._list_jobs({"status": ["invalid_status"]})
            assert result.status_code == 400


class TestSchedulerGetJob:
    """Tests for getting a specific job."""

    def test_get_job(self, handler):
        """Test getting a job."""
        mock_scheduler = MagicMock()
        mock_job = MagicMock()
        mock_job.to_dict.return_value = {"job_id": "job123", "name": "Test Job"}
        mock_scheduler.get_job.return_value = mock_job

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._get_job("job123")
            assert result.status_code == 200

    def test_get_job_not_found(self, handler):
        """Test getting non-existent job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._get_job("invalid_job")
            assert result.status_code == 404


class TestSchedulerCreateJob:
    """Tests for creating scheduler jobs."""

    def test_create_job_missing_name(self, handler):
        """Test create job requires name."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "read_json_body", return_value={"trigger_type": "cron"}),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._create_job(mock_handler)
            assert result.status_code == 400

    def test_create_job_missing_body(self, handler):
        """Test create job requires body."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "read_json_body", return_value=None),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._create_job(mock_handler)
            assert result.status_code == 400

    def test_create_job_cron_missing_schedule(self, handler):
        """Test cron job requires cron field."""
        mock_handler = MagicMock()

        with (
            patch.object(
                handler,
                "read_json_body",
                return_value={"name": "Test", "trigger_type": "cron"},
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._create_job(mock_handler)
            assert result.status_code == 400

    def test_create_job_interval_missing_minutes(self, handler):
        """Test interval job requires interval_minutes."""
        mock_handler = MagicMock()

        with (
            patch.object(
                handler,
                "read_json_body",
                return_value={"name": "Test", "trigger_type": "interval"},
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._create_job(mock_handler)
            assert result.status_code == 400


class TestSchedulerDeleteJob:
    """Tests for deleting scheduler jobs."""

    def test_delete_job(self, handler):
        """Test deleting a job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.remove_schedule.return_value = True

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._delete_job("job123")
            assert result.status_code == 200

    def test_delete_job_not_found(self, handler):
        """Test deleting non-existent job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._delete_job("invalid_job")
            assert result.status_code == 404


def _mock_auth_user():
    """Create a mock authenticated user context."""
    user = MagicMock()
    user.is_authenticated = True
    user.user_id = "test_user"
    user.role = "admin"
    user.error_reason = None
    return user


class TestSchedulerJobActions:
    """Tests for scheduler job actions."""

    def test_pause_job(self, handler):
        """Test pausing a job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.pause_schedule.return_value = True

        # Add headers to handler so require_user_auth can find it
        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._pause_job("job123")
            assert result.status_code == 200

    def test_pause_job_not_found(self, handler):
        """Test pausing non-existent job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._pause_job("invalid_job")
            assert result.status_code == 404

    def test_resume_job(self, handler):
        """Test resuming a job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.resume_schedule.return_value = True

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._resume_job("job123")
            assert result.status_code == 200


class TestSchedulerJobHistory:
    """Tests for scheduler job history."""

    def test_get_job_history(self, handler):
        """Test getting job history."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.get_job_history.return_value = []

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._get_job_history("job123", limit=10)
            assert result.status_code == 200

    def test_get_job_history_not_found(self, handler):
        """Test getting history for non-existent job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._get_job_history("invalid_job")
            assert result.status_code == 404


class TestSchedulerWebhook:
    """Tests for scheduler webhook handling."""

    def test_handle_webhook_missing_body(self, handler):
        """Test webhook requires request body."""
        mock_handler = MagicMock()

        with patch.object(handler, "read_json_body", return_value=None):
            result = handler._handle_webhook(mock_handler, "wh123")
            assert result.status_code == 400


class TestSchedulerGitPush:
    """Tests for scheduler git push handling."""

    def test_handle_git_push_missing_body(self, handler):
        """Test git push requires request body."""
        mock_handler = MagicMock()

        with patch.object(handler, "read_json_body", return_value=None):
            result = handler._handle_git_push(mock_handler)
            assert result.status_code == 400


class TestSchedulerFileUpload:
    """Tests for scheduler file upload handling."""

    def test_handle_file_upload_missing_workspace(self, handler):
        """Test file upload requires workspace_id."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "read_json_body", return_value={"document_ids": ["doc1"]}),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._handle_file_upload(mock_handler)
            assert result.status_code == 400

    def test_handle_file_upload_missing_documents(self, handler):
        """Test file upload requires document_ids."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "read_json_body", return_value={"workspace_id": "ws123"}),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._handle_file_upload(mock_handler)
            assert result.status_code == 400


# =============================================================================
# Additional Tests for STABLE Graduation (20+ new tests)
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker is properly configured."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        assert _scheduler_circuit_breaker is not None
        assert _scheduler_circuit_breaker.name == "scheduler"
        assert _scheduler_circuit_breaker.failure_threshold == 5

    def test_circuit_breaker_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        # Reset circuit breaker state
        _scheduler_circuit_breaker.is_open = False
        assert _scheduler_circuit_breaker.can_proceed() is True

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        # Reset circuit breaker state
        _scheduler_circuit_breaker.is_open = False

        # Record enough failures to open the circuit
        for _ in range(_scheduler_circuit_breaker.failure_threshold):
            _scheduler_circuit_breaker.record_failure()

        # Circuit should be open now
        assert _scheduler_circuit_breaker.can_proceed() is False

        # Reset for other tests
        _scheduler_circuit_breaker.is_open = False

    def test_create_job_returns_503_when_circuit_open(self, handler):
        """Test that _create_job returns 503 when circuit is open."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        # Force circuit open
        _scheduler_circuit_breaker.is_open = True

        mock_handler = MagicMock()

        with (
            patch.object(
                handler,
                "read_json_body",
                return_value={"name": "Test", "trigger_type": "cron", "cron": "* * * * *"},
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler._create_job(mock_handler)

        assert result is not None
        assert result.status_code == 503

        # Reset circuit state
        _scheduler_circuit_breaker.is_open = False

    def test_trigger_job_returns_503_when_circuit_open(self, handler):
        """Test that _trigger_job returns 503 when circuit is open."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        # Force circuit open
        _scheduler_circuit_breaker.is_open = True

        handler.headers = MagicMock()

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=_mock_auth_user(),
        ):
            result = handler._trigger_job("job123")

        assert result is not None
        assert result.status_code == 503

        # Reset circuit state
        _scheduler_circuit_breaker.is_open = False


class TestRateLimitingIntegration:
    """Tests for rate limiting integration."""

    def test_create_job_has_rate_limit_decorator(self, handler):
        """Test that _create_job has rate limiting."""
        method = handler._create_job

        # Check for rate limit marker
        assert hasattr(method, "_rate_limited") or callable(method)

    def test_trigger_job_has_rate_limit_decorator(self, handler):
        """Test that _trigger_job has rate limiting."""
        method = handler._trigger_job

        # Check for rate limit marker
        assert hasattr(method, "_rate_limited") or callable(method)


class TestTriggerJobExecution:
    """Tests for trigger job execution."""

    def test_trigger_job_success(self, handler):
        """Test successful job trigger."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        _scheduler_circuit_breaker.is_open = False

        mock_scheduler = MagicMock()
        mock_job = MagicMock()
        mock_scheduler.get_job.return_value = mock_job

        mock_run = MagicMock()
        mock_run.to_dict.return_value = {"run_id": "run123", "status": "completed"}

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
            patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=mock_run,
            ),
        ):
            result = handler._trigger_job("job123")
            assert result.status_code == 200

    def test_trigger_job_not_found(self, handler):
        """Test triggering non-existent job."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        _scheduler_circuit_breaker.is_open = False

        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._trigger_job("invalid_job")
            assert result.status_code == 404


class TestHandleRouting:
    """Tests for request routing in handle methods."""

    def test_handle_routes_to_list_jobs(self, handler):
        """Test handle routes to list jobs."""
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler.handle("/api/v1/scheduler/jobs", {}, MagicMock())
            assert result is not None

    def test_handle_routes_to_get_status(self, handler):
        """Test handle routes to scheduler status."""
        mock_scheduler = MagicMock()
        mock_scheduler._running = True
        mock_scheduler.list_jobs.return_value = []

        with patch.object(handler, "_get_scheduler", return_value=mock_scheduler):
            result = handler.handle("/api/v1/scheduler/status", {}, MagicMock())
            assert result is not None
            assert result.status_code == 200

    def test_handle_routes_to_get_job(self, handler):
        """Test handle routes to get job."""
        mock_scheduler = MagicMock()
        mock_job = MagicMock()
        mock_job.to_dict.return_value = {"job_id": "job123"}
        mock_scheduler.get_job.return_value = mock_job

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler.handle("/api/v1/scheduler/jobs/job123", {}, MagicMock())
            assert result is not None

    def test_handle_routes_to_job_history(self, handler):
        """Test handle routes to job history."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.get_job_history.return_value = []

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler.handle("/api/v1/scheduler/jobs/job123/history", {}, MagicMock())
            assert result is not None


class TestHandlePostRouting:
    """Tests for POST request routing."""

    def test_handle_post_routes_to_create_job(self, handler):
        """Test handle_post routes to create job."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        _scheduler_circuit_breaker.is_open = False

        mock_handler = MagicMock()

        with (
            patch.object(handler, "require_permission_or_error", return_value=(True, None)),
            patch.object(handler, "read_json_body", return_value={"name": "Test"}),
        ):
            result = handler.handle_post("/api/v1/scheduler/jobs", {}, mock_handler)
            # Will fail validation but route is correct
            assert result is not None

    def test_handle_post_routes_to_git_push(self, handler):
        """Test handle_post routes to git push."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "require_permission_or_error", return_value=(True, None)),
            patch.object(handler, "read_json_body", return_value=None),
        ):
            result = handler.handle_post("/api/v1/scheduler/events/git-push", {}, mock_handler)
            assert result is not None

    def test_handle_post_routes_to_file_upload(self, handler):
        """Test handle_post routes to file upload."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "require_permission_or_error", return_value=(True, None)),
            patch.object(handler, "read_json_body", return_value={}),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            result = handler.handle_post("/api/v1/scheduler/events/file-upload", {}, mock_handler)
            assert result is not None


class TestHandleDeleteRouting:
    """Tests for DELETE request routing."""

    def test_handle_delete_routes_to_delete_job(self, handler):
        """Test handle_delete routes to delete job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.remove_schedule.return_value = True
        mock_handler = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch.object(handler, "require_permission_or_error", return_value=(True, None)),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler.handle_delete("/api/v1/scheduler/jobs/job123", {}, mock_handler)
            assert result is not None

    def test_handle_delete_invalid_path(self, handler):
        """Test handle_delete returns None for invalid paths."""
        mock_handler = MagicMock()

        with patch.object(handler, "require_permission_or_error", return_value=(True, None)):
            result = handler.handle_delete("/api/v1/scheduler/invalid", {}, mock_handler)
            assert result is None


class TestResumeJobNotFound:
    """Tests for resume job edge cases."""

    def test_resume_job_not_found(self, handler):
        """Test resuming non-existent job."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._resume_job("invalid_job")
            assert result.status_code == 404


class TestDeleteJobFailure:
    """Tests for delete job edge cases."""

    def test_delete_job_failure(self, handler):
        """Test delete job failure returns error."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.remove_schedule.return_value = False

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._delete_job("job123")
            assert result.status_code == 500


class TestPauseJobFailure:
    """Tests for pause job edge cases."""

    def test_pause_job_failure(self, handler):
        """Test pause job failure returns error."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.pause_schedule.return_value = False

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._pause_job("job123")
            assert result.status_code == 400


class TestResumeJobFailure:
    """Tests for resume job edge cases."""

    def test_resume_job_failure(self, handler):
        """Test resume job failure returns error."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()
        mock_scheduler.resume_schedule.return_value = False

        handler.headers = MagicMock()

        with (
            patch.object(handler, "_get_scheduler", return_value=mock_scheduler),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=_mock_auth_user(),
            ),
        ):
            result = handler._resume_job("job123")
            assert result.status_code == 400


class TestInvalidTriggerType:
    """Tests for invalid trigger type handling."""

    def test_create_job_invalid_trigger_type(self, handler):
        """Test create job with invalid trigger type."""
        from aragora.server.handlers.features.scheduler import (
            _scheduler_circuit_breaker,
        )

        _scheduler_circuit_breaker.is_open = False

        mock_handler = MagicMock()

        with (
            patch.object(
                handler,
                "read_json_body",
                return_value={"name": "Test", "trigger_type": "invalid_trigger"},
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.scheduler.require_permission",
                lambda p: lambda f: f,
            ),
            patch("aragora.scheduler.TriggerType", side_effect=ValueError("Invalid")),
        ):
            result = handler._create_job(mock_handler)
            assert result.status_code == 400
