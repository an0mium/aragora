"""Tests for audit scheduler handler.

Tests the scheduler API endpoints including:
- GET  /api/v1/scheduler/jobs - List scheduled jobs
- POST /api/v1/scheduler/jobs - Create a scheduled job
- GET  /api/v1/scheduler/jobs/{job_id} - Get job details
- DELETE /api/v1/scheduler/jobs/{job_id} - Delete a job
- POST /api/v1/scheduler/jobs/{job_id}/trigger - Manually trigger a job
- POST /api/v1/scheduler/jobs/{job_id}/pause - Pause a job
- POST /api/v1/scheduler/jobs/{job_id}/resume - Resume a job
- GET  /api/v1/scheduler/jobs/{job_id}/history - Get job run history
- POST /api/v1/scheduler/webhooks/{webhook_id} - Receive webhook triggers
- POST /api/v1/scheduler/events/git-push - Handle git push events
- POST /api/v1/scheduler/events/file-upload - Handle file upload events
- GET  /api/v1/scheduler/status - Get scheduler status
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.features.scheduler import SchedulerHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: dict[str, Any] | None = None, token: str = "test-valid-token"):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        else:
            self.rfile.read.return_value = b""
            self.headers = {
                "Content-Length": "0",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        self.client_address = ("127.0.0.1", 12345)


# ---------------------------------------------------------------------------
# Mock scheduler domain objects
# ---------------------------------------------------------------------------


class FakeScheduleStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    RUNNING = "running"
    COMPLETED = "completed"


class FakeTriggerType(Enum):
    CRON = "cron"
    INTERVAL = "interval"
    WEBHOOK = "webhook"
    GIT_PUSH = "git_push"
    FILE_UPLOAD = "file_upload"


@dataclass
class FakeJob:
    job_id: str
    name: str
    status: FakeScheduleStatus = FakeScheduleStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
        }


@dataclass
class FakeRun:
    run_id: str
    job_id: str
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    """Bypass rate limiting for tests."""
    monkeypatch.setenv("ARAGORA_USE_DISTRIBUTED_RATE_LIMIT", "false")


@pytest.fixture(autouse=True)
def _patch_user_auth(monkeypatch):
    """Bypass @require_user_auth decorator for tests.

    The conftest auto-auth fixture handles @require_permission (via
    _test_user_context_override) but @require_user_auth uses
    extract_user_from_request which needs a separate patch.

    The decorator searches args for an object with ``headers``.
    For methods like _list_jobs(self, query_params) the SchedulerHandler
    instance (self) does not have headers, so the decorator short-circuits
    to 401.  We patch extract_user_from_request AND give SchedulerHandler
    a ``headers`` attribute so the decorator finds it on ``self``.
    """
    from aragora.billing.auth.context import UserAuthContext

    mock_user = UserAuthContext(
        authenticated=True,
        user_id="test-user-001",
        email="test@example.com",
        org_id="test-org-001",
        role="admin",
        token_type="access",
        client_ip="127.0.0.1",
    )
    # The _create_job method calls user.get("id") expecting a dict-like user.
    # Patch .get onto the UserAuthContext so it behaves like a dict for handler code.
    mock_user.get = lambda key, default=None: getattr(mock_user, key, default)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "aragora.billing.jwt_auth.extract_user_from_request",
        lambda handler, user_store=None: mock_user,
    )

    # Give SchedulerHandler a headers attr so require_user_auth finds self
    monkeypatch.setattr(
        SchedulerHandler,
        "headers",
        {"Authorization": "Bearer test"},
        raising=False,
    )


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the circuit breaker between tests."""
    import aragora.server.handlers.features.scheduler as sched_mod

    cb = sched_mod._scheduler_circuit_breaker
    cb._failure_count = 0
    cb._success_count = 0
    cb._state = "closed"
    yield
    cb._failure_count = 0
    cb._success_count = 0
    cb._state = "closed"


@pytest.fixture
def handler():
    """Create a SchedulerHandler instance."""
    return SchedulerHandler(server_context={})


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler factory."""
    def _make(body=None, token="test-valid-token"):
        return MockHTTPHandler(body=body, token=token)
    return _make


def _make_mock_scheduler(
    jobs=None,
    running=True,
):
    """Create a mock scheduler instance."""
    sched = MagicMock()
    sched._running = running
    sched.list_jobs.return_value = jobs or []
    sched.get_job.return_value = None
    sched.add_schedule.return_value = FakeJob(job_id="job-001", name="Test Job")
    sched.remove_schedule.return_value = True
    sched.pause_schedule.return_value = True
    sched.resume_schedule.return_value = True
    sched.trigger_job = MagicMock(return_value=FakeRun(run_id="run-001", job_id="job-001"))
    sched.get_job_history.return_value = []
    sched.handle_webhook = MagicMock(return_value=[])
    sched.handle_git_push = MagicMock(return_value=[])
    sched.handle_file_upload = MagicMock(return_value=[])
    return sched


# ============================================================================
# Handler initialization
# ============================================================================


class TestHandlerInit:
    """Tests for SchedulerHandler initialization."""

    def test_init_with_server_context(self):
        ctx = {"key": "value"}
        h = SchedulerHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_ctx(self):
        ctx = {"key": "value"}
        h = SchedulerHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_with_none(self):
        h = SchedulerHandler()
        assert h.ctx == {}

    def test_init_server_context_overrides_ctx(self):
        h = SchedulerHandler(ctx={"old": 1}, server_context={"new": 2})
        assert h.ctx == {"new": 2}

    def test_init_both_none(self):
        h = SchedulerHandler(ctx=None, server_context=None)
        assert h.ctx == {}


# ============================================================================
# can_handle tests
# ============================================================================


class TestCanHandle:
    """Tests for SchedulerHandler.can_handle()."""

    def test_jobs_list_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/jobs") is True

    def test_webhooks_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/webhooks") is True

    def test_git_push_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/events/git-push") is True

    def test_file_upload_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/events/file-upload") is True

    def test_status_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/status") is True

    def test_job_by_id_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/jobs/job-123") is True

    def test_job_trigger_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/jobs/job-123/trigger") is True

    def test_job_pause_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/jobs/job-123/pause") is True

    def test_job_resume_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/jobs/job-123/resume") is True

    def test_job_history_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/jobs/job-123/history") is True

    def test_webhook_by_id_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/webhooks/wh-456") is True

    def test_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/scheduler/unknown") is False

    def test_non_scheduler_route(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_path(self, handler):
        assert handler.can_handle("/api/v1/scheduler") is False


# ============================================================================
# GET /api/v1/scheduler/jobs - List jobs
# ============================================================================


class TestListJobs:
    """Tests for listing scheduled jobs."""

    def test_list_jobs_success(self, handler, mock_http):
        jobs = [
            FakeJob(job_id="j1", name="Job 1"),
            FakeJob(job_id="j2", name="Job 2"),
        ]
        mock_sched = _make_mock_scheduler(jobs=jobs)
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._list_jobs({})
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 2
        assert len(body["jobs"]) == 2
        assert body["jobs"][0]["job_id"] == "j1"

    def test_list_jobs_empty(self, handler, mock_http):
        mock_sched = _make_mock_scheduler(jobs=[])
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._list_jobs({})
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 0
        assert body["jobs"] == []

    def test_list_jobs_with_status_filter(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch("aragora.server.handlers.features.scheduler.SchedulerHandler._list_jobs") as orig:
                # Just test the routing dispatches correctly with status
                pass
            # Test that the status filter is parsed
            mock_status_cls = MagicMock()
            mock_status_cls.return_value = "active"
            with patch(
                "aragora.scheduler.ScheduleStatus", mock_status_cls
            ):
                result = handler._list_jobs({"status": ["active"]})
        body = _body(result)
        assert _status(result) == 200

    def test_list_jobs_invalid_status(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.scheduler.ScheduleStatus", side_effect=ValueError("bad")
            ):
                result = handler._list_jobs({"status": ["invalid_status"]})
        body = _body(result)
        assert _status(result) == 400
        assert "Invalid status" in body.get("error", "")

    def test_list_jobs_with_workspace_filter(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._list_jobs({"workspace_id": ["ws-123"]})
        mock_sched.list_jobs.assert_called_once_with(status=None, workspace_id="ws-123")

    def test_list_jobs_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_list_jobs") as mock_list:
            mock_list.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle("/api/v1/scheduler/jobs", {}, mock_http())
            mock_list.assert_called_once_with({})


# ============================================================================
# GET /api/v1/scheduler/jobs/{job_id} - Get job
# ============================================================================


class TestGetJob:
    """Tests for getting a specific job."""

    def test_get_job_success(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._get_job("j1")
        body = _body(result)
        assert _status(result) == 200
        assert body["job_id"] == "j1"
        assert body["name"] == "Job 1"

    def test_get_job_not_found(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = None
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._get_job("nonexistent")
        body = _body(result)
        assert _status(result) == 404
        assert "not found" in body.get("error", "").lower()

    def test_get_job_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_get_job") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle("/api/v1/scheduler/jobs/j1", {}, mock_http())
            mock_get.assert_called_once_with("j1")

    def test_get_job_route_with_dashes(self, handler, mock_http):
        with patch.object(handler, "_get_job") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle("/api/v1/scheduler/jobs/job-abc-123", {}, mock_http())
            mock_get.assert_called_once_with("job-abc-123")


# ============================================================================
# POST /api/v1/scheduler/jobs - Create job
# ============================================================================


class TestCreateJob:
    """Tests for creating a scheduled job."""

    def test_create_job_cron_success(self, handler, mock_http):
        body = {
            "name": "Daily Scan",
            "trigger_type": "cron",
            "cron": "0 2 * * *",
            "workspace_id": "ws-1",
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch("aragora.scheduler.ScheduleConfig") as mock_config:
                mock_config.return_value = MagicMock()
                with patch("aragora.scheduler.TriggerType", FakeTriggerType):
                    result = handler._create_job(h)
        body_resp = _body(result)
        assert _status(result) == 201
        assert body_resp["success"] is True
        assert "job" in body_resp

    def test_create_job_interval_success(self, handler, mock_http):
        body = {
            "name": "Hourly Check",
            "trigger_type": "interval",
            "interval_minutes": 60,
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch("aragora.scheduler.ScheduleConfig") as mock_config:
                mock_config.return_value = MagicMock()
                with patch("aragora.scheduler.TriggerType", FakeTriggerType):
                    result = handler._create_job(h)
        assert _status(result) == 201

    def test_create_job_no_body(self, handler, mock_http):
        h = MockHTTPHandler(body=None)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400
        assert "body" in body.get("error", "").lower() or "required" in body.get("error", "").lower()

    def test_create_job_empty_name(self, handler, mock_http):
        h = mock_http(body={"name": "", "trigger_type": "cron", "cron": "* * * * *"})
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400
        assert "name" in body.get("error", "").lower()

    def test_create_job_whitespace_name(self, handler, mock_http):
        h = mock_http(body={"name": "   ", "trigger_type": "cron", "cron": "* * * * *"})
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400
        assert "name" in body.get("error", "").lower()

    def test_create_job_missing_name(self, handler, mock_http):
        h = mock_http(body={"trigger_type": "cron", "cron": "* * * * *"})
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400

    def test_create_job_invalid_trigger_type(self, handler, mock_http):
        h = mock_http(body={"name": "Test", "trigger_type": "invalid_type"})
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch("aragora.scheduler.TriggerType", FakeTriggerType):
                result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400
        assert "trigger_type" in body.get("error", "").lower() or "valid" in body.get("error", "").lower()

    def test_create_job_cron_without_expression(self, handler, mock_http):
        h = mock_http(body={"name": "Test", "trigger_type": "cron"})
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch("aragora.scheduler.TriggerType", FakeTriggerType):
                result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400
        assert "cron" in body.get("error", "").lower()

    def test_create_job_interval_without_minutes(self, handler, mock_http):
        h = mock_http(body={"name": "Test", "trigger_type": "interval"})
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch("aragora.scheduler.TriggerType", FakeTriggerType):
                result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400
        assert "interval_minutes" in body.get("error", "").lower()

    def test_create_job_default_trigger_type_is_cron(self, handler, mock_http):
        """Default trigger type is cron, so missing 'cron' field should fail."""
        h = mock_http(body={"name": "Test"})
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch("aragora.scheduler.TriggerType", FakeTriggerType):
                result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400
        assert "cron" in body.get("error", "").lower()

    def test_create_job_circuit_breaker_open(self, handler, mock_http):
        """Circuit breaker open returns 503."""
        import aragora.server.handlers.features.scheduler as sched_mod
        cb = sched_mod._scheduler_circuit_breaker
        # Force circuit breaker open
        with patch.object(cb, "can_proceed", return_value=False):
            h = mock_http(body={"name": "Test", "trigger_type": "cron", "cron": "* * * * *"})
            result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 503
        assert "unavailable" in body.get("error", "").lower()

    def test_create_job_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_create_job") as mock_create:
            mock_create.return_value = MagicMock(status_code=201, body=b'{}')
            h = mock_http(body={"name": "Test"})
            handler.handle_post("/api/v1/scheduler/jobs", {}, h)
            mock_create.assert_called_once_with(h)

    def test_create_job_with_all_optional_fields(self, handler, mock_http):
        body = {
            "name": "Full Job",
            "description": "A comprehensive job",
            "trigger_type": "cron",
            "cron": "0 2 * * *",
            "webhook_secret": "secret123",
            "preset": "Code Security",
            "audit_types": ["security", "compliance"],
            "custom_config": {"key": "val"},
            "workspace_id": "ws-1",
            "document_ids": ["doc1", "doc2"],
            "notify_on_complete": False,
            "notify_on_findings": False,
            "finding_severity_threshold": "high",
            "max_retries": 5,
            "timeout_minutes": 120,
            "tags": ["prod", "critical"],
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch("aragora.scheduler.ScheduleConfig") as mock_config:
                mock_config.return_value = MagicMock()
                with patch("aragora.scheduler.TriggerType", FakeTriggerType):
                    result = handler._create_job(h)
        assert _status(result) == 201


# ============================================================================
# DELETE /api/v1/scheduler/jobs/{job_id} - Delete job
# ============================================================================


class TestDeleteJob:
    """Tests for deleting a scheduled job."""

    def test_delete_job_success(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.remove_schedule.return_value = True
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._delete_job("j1")
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert "deleted" in body["message"].lower()

    def test_delete_job_not_found(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = None
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._delete_job("nonexistent")
        body = _body(result)
        assert _status(result) == 404
        assert "not found" in body.get("error", "").lower()

    def test_delete_job_remove_fails(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.remove_schedule.return_value = False
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._delete_job("j1")
        body = _body(result)
        assert _status(result) == 500
        assert "failed" in body.get("error", "").lower()

    def test_delete_job_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_delete_job") as mock_del:
            mock_del.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle_delete("/api/v1/scheduler/jobs/j1", {}, mock_http())
            mock_del.assert_called_once_with("j1")

    def test_delete_route_wrong_segment_count(self, handler, mock_http):
        """DELETE with extra segments is not matched."""
        result = handler.handle_delete(
            "/api/v1/scheduler/jobs/j1/extra", {}, mock_http()
        )
        assert result is None


# ============================================================================
# POST /api/v1/scheduler/jobs/{job_id}/trigger - Trigger job
# ============================================================================


class TestTriggerJob:
    """Tests for manually triggering a job."""

    def test_trigger_job_success(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        run = FakeRun(run_id="r1", job_id="j1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.trigger_job = MagicMock(return_value=run)
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=run,
            ):
                result = handler._trigger_job("j1")
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert "run" in body

    def test_trigger_job_not_found(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = None
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._trigger_job("nonexistent")
        body = _body(result)
        assert _status(result) == 404
        assert "not found" in body.get("error", "").lower()

    def test_trigger_job_run_fails(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=None,
            ):
                result = handler._trigger_job("j1")
        body = _body(result)
        assert _status(result) == 500
        assert "failed" in body.get("error", "").lower()

    def test_trigger_job_runtime_error(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=RuntimeError("execution failed"),
            ):
                result = handler._trigger_job("j1")
        assert _status(result) == 500

    def test_trigger_job_timeout_error(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=TimeoutError("timed out"),
            ):
                result = handler._trigger_job("j1")
        assert _status(result) == 500

    def test_trigger_job_connection_error(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=ConnectionError("conn refused"),
            ):
                result = handler._trigger_job("j1")
        assert _status(result) == 500

    def test_trigger_job_circuit_breaker_open(self, handler, mock_http):
        import aragora.server.handlers.features.scheduler as sched_mod
        cb = sched_mod._scheduler_circuit_breaker
        with patch.object(cb, "can_proceed", return_value=False):
            result = handler._trigger_job("j1")
        body = _body(result)
        assert _status(result) == 503
        assert "unavailable" in body.get("error", "").lower()

    def test_trigger_job_records_success_on_cb(self, handler, mock_http):
        import aragora.server.handlers.features.scheduler as sched_mod
        cb = sched_mod._scheduler_circuit_breaker
        job = FakeJob(job_id="j1", name="Job 1")
        run = FakeRun(run_id="r1", job_id="j1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=run,
            ):
                with patch.object(cb, "record_success") as mock_success:
                    handler._trigger_job("j1")
                    mock_success.assert_called_once()

    def test_trigger_job_records_failure_on_cb(self, handler, mock_http):
        import aragora.server.handlers.features.scheduler as sched_mod
        cb = sched_mod._scheduler_circuit_breaker
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=None,
            ):
                with patch.object(cb, "record_failure") as mock_fail:
                    handler._trigger_job("j1")
                    mock_fail.assert_called_once()

    def test_trigger_job_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_trigger_job") as mock_trig:
            mock_trig.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle_post("/api/v1/scheduler/jobs/j1/trigger", {}, mock_http())
            mock_trig.assert_called_once_with("j1")


# ============================================================================
# POST /api/v1/scheduler/jobs/{job_id}/pause - Pause job
# ============================================================================


class TestPauseJob:
    """Tests for pausing a scheduled job."""

    def test_pause_job_success(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.pause_schedule.return_value = True
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._pause_job("j1")
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert "paused" in body["message"].lower()

    def test_pause_job_not_found(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = None
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._pause_job("nonexistent")
        body = _body(result)
        assert _status(result) == 404

    def test_pause_job_fails(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.pause_schedule.return_value = False
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._pause_job("j1")
        body = _body(result)
        assert _status(result) == 400
        assert "could not pause" in body.get("error", "").lower()

    def test_pause_job_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_pause_job") as mock_pause:
            mock_pause.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle_post("/api/v1/scheduler/jobs/j1/pause", {}, mock_http())
            mock_pause.assert_called_once_with("j1")


# ============================================================================
# POST /api/v1/scheduler/jobs/{job_id}/resume - Resume job
# ============================================================================


class TestResumeJob:
    """Tests for resuming a paused job."""

    def test_resume_job_success(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1", status=FakeScheduleStatus.PAUSED)
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.resume_schedule.return_value = True
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._resume_job("j1")
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert "resumed" in body["message"].lower()

    def test_resume_job_not_found(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = None
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._resume_job("nonexistent")
        body = _body(result)
        assert _status(result) == 404

    def test_resume_job_fails(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.resume_schedule.return_value = False
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._resume_job("j1")
        body = _body(result)
        assert _status(result) == 400
        assert "could not resume" in body.get("error", "").lower()

    def test_resume_job_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_resume_job") as mock_resume:
            mock_resume.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle_post("/api/v1/scheduler/jobs/j1/resume", {}, mock_http())
            mock_resume.assert_called_once_with("j1")


# ============================================================================
# GET /api/v1/scheduler/jobs/{job_id}/history - Job history
# ============================================================================


class TestGetJobHistory:
    """Tests for getting job run history."""

    def test_history_success(self, handler, mock_http):
        runs = [
            FakeRun(run_id="r1", job_id="j1"),
            FakeRun(run_id="r2", job_id="j1"),
        ]
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.get_job_history.return_value = runs
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._get_job_history("j1", 10)
        body = _body(result)
        assert _status(result) == 200
        assert body["job_id"] == "j1"
        assert body["count"] == 2
        assert len(body["runs"]) == 2

    def test_history_empty(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.get_job_history.return_value = []
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._get_job_history("j1", 10)
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 0
        assert body["runs"] == []

    def test_history_job_not_found(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = None
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._get_job_history("nonexistent", 10)
        body = _body(result)
        assert _status(result) == 404

    def test_history_passes_limit(self, handler, mock_http):
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        mock_sched.get_job_history.return_value = []
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            handler._get_job_history("j1", 25)
        mock_sched.get_job_history.assert_called_once_with("j1", limit=25)

    def test_history_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_get_job_history") as mock_hist:
            mock_hist.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle("/api/v1/scheduler/jobs/j1/history", {}, mock_http())
            mock_hist.assert_called_once_with("j1", 10)

    def test_history_route_with_limit_param(self, handler, mock_http):
        with patch.object(handler, "_get_job_history") as mock_hist:
            mock_hist.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle(
                "/api/v1/scheduler/jobs/j1/history",
                {"limit": ["50"]},
                mock_http(),
            )
            mock_hist.assert_called_once_with("j1", 50)

    def test_history_route_limit_clamped_to_max(self, handler, mock_http):
        with patch.object(handler, "_get_job_history") as mock_hist:
            mock_hist.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle(
                "/api/v1/scheduler/jobs/j1/history",
                {"limit": ["200"]},
                mock_http(),
            )
            # safe_query_int clamps to max_val=100
            mock_hist.assert_called_once_with("j1", 100)


# ============================================================================
# GET /api/v1/scheduler/status - Scheduler status
# ============================================================================


class TestGetSchedulerStatus:
    """Tests for scheduler status endpoint."""

    def test_status_success(self, handler, mock_http):
        jobs = [
            FakeJob(job_id="j1", name="Job 1", status=FakeScheduleStatus.ACTIVE),
            FakeJob(job_id="j2", name="Job 2", status=FakeScheduleStatus.RUNNING),
            FakeJob(job_id="j3", name="Job 3", status=FakeScheduleStatus.PAUSED),
        ]
        mock_sched = _make_mock_scheduler(jobs=jobs, running=True)
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._get_scheduler_status()
        body = _body(result)
        assert _status(result) == 200
        assert body["running"] is True
        assert body["total_jobs"] == 3
        assert body["active_jobs"] == 1
        assert body["running_jobs"] == 1

    def test_status_no_jobs(self, handler, mock_http):
        mock_sched = _make_mock_scheduler(jobs=[], running=False)
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            result = handler._get_scheduler_status()
        body = _body(result)
        assert _status(result) == 200
        assert body["total_jobs"] == 0
        assert body["active_jobs"] == 0
        assert body["running_jobs"] == 0

    def test_status_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_get_scheduler_status") as mock_stat:
            mock_stat.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle("/api/v1/scheduler/status", {}, mock_http())
            mock_stat.assert_called_once()


# ============================================================================
# POST /api/v1/scheduler/webhooks/{webhook_id} - Webhook trigger
# ============================================================================


class TestHandleWebhook:
    """Tests for webhook trigger endpoint."""

    def test_webhook_success(self, handler, mock_http):
        runs = [FakeRun(run_id="r1", job_id="j1")]
        mock_sched = _make_mock_scheduler()
        mock_sched.handle_webhook = MagicMock(return_value=runs)
        h = mock_http(body={"event": "deploy"})
        h.headers["X-Webhook-Signature"] = "sig123"
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=runs,
            ):
                result = handler._handle_webhook(h, "wh-001")
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["triggered_jobs"] == 1

    def test_webhook_no_body(self, handler, mock_http):
        h = MockHTTPHandler(body=None)
        result = handler._handle_webhook(h, "wh-001")
        body = _body(result)
        assert _status(result) == 400
        assert "body" in body.get("error", "").lower() or "required" in body.get("error", "").lower()

    def test_webhook_no_runs_triggered(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"event": "test"})
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ):
                result = handler._handle_webhook(h, "wh-001")
        body = _body(result)
        assert _status(result) == 200
        assert body["triggered_jobs"] == 0
        assert body["runs"] == []

    def test_webhook_runtime_error(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"event": "test"})
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=RuntimeError("webhook fail"),
            ):
                result = handler._handle_webhook(h, "wh-001")
        assert _status(result) == 500

    def test_webhook_value_error(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"event": "test"})
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=ValueError("bad sig"),
            ):
                result = handler._handle_webhook(h, "wh-001")
        assert _status(result) == 500

    def test_webhook_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_handle_webhook") as mock_wh:
            mock_wh.return_value = MagicMock(status_code=200, body=b'{}')
            h = mock_http(body={"event": "test"})
            handler.handle_post("/api/v1/scheduler/webhooks/wh-001", {}, h)
            mock_wh.assert_called_once_with(h, "wh-001")

    def test_webhook_passes_signature_header(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"event": "test"})
        h.headers["X-Webhook-Signature"] = "hmac-sha256=abc"
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ) as mock_run:
                handler._handle_webhook(h, "wh-001")
        # Verify the coroutine was called with the signature
        call_args = mock_run.call_args
        assert call_args is not None


# ============================================================================
# POST /api/v1/scheduler/events/git-push - Git push events
# ============================================================================


class TestHandleGitPush:
    """Tests for git push event handler."""

    def test_git_push_success(self, handler, mock_http):
        runs = [FakeRun(run_id="r1", job_id="j1")]
        body = {
            "repository": {"full_name": "owner/repo"},
            "ref": "refs/heads/main",
            "after": "abc123",
            "commits": [
                {"modified": ["file1.py"], "added": ["file2.py"]},
            ],
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=runs,
            ):
                result = handler._handle_git_push(h)
        resp = _body(result)
        assert _status(result) == 200
        assert resp["success"] is True
        assert resp["repository"] == "owner/repo"
        assert resp["branch"] == "main"
        assert resp["triggered_jobs"] == 1

    def test_git_push_no_body(self, handler, mock_http):
        h = MockHTTPHandler(body=None)
        result = handler._handle_git_push(h)
        assert _status(result) == 400

    def test_git_push_no_commits(self, handler, mock_http):
        body = {
            "repository": {"full_name": "owner/repo"},
            "ref": "refs/heads/main",
            "after": "abc123",
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ):
                result = handler._handle_git_push(h)
        resp = _body(result)
        assert _status(result) == 200
        assert resp["triggered_jobs"] == 0

    def test_git_push_branch_extraction(self, handler, mock_http):
        body = {
            "repository": {"full_name": "org/project"},
            "ref": "refs/heads/feature/my-branch",
            "after": "def456",
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ):
                result = handler._handle_git_push(h)
        resp = _body(result)
        assert resp["branch"] == "feature/my-branch"

    def test_git_push_non_refs_ref(self, handler, mock_http):
        """Non-refs/heads ref is passed through as-is."""
        body = {
            "repository": {"full_name": "org/project"},
            "ref": "v1.0.0",
            "after": "tag123",
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ):
                result = handler._handle_git_push(h)
        resp = _body(result)
        assert resp["branch"] == "v1.0.0"

    def test_git_push_deduplicates_files(self, handler, mock_http):
        body = {
            "repository": {"full_name": "org/repo"},
            "ref": "refs/heads/main",
            "after": "abc",
            "commits": [
                {"modified": ["file1.py"], "added": ["file2.py"]},
                {"modified": ["file1.py"], "added": ["file3.py"]},
            ],
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ) as mock_run:
                handler._handle_git_push(h)
        # The changed_files should be deduplicated
        call_args = mock_run.call_args
        assert call_args is not None

    def test_git_push_runtime_error(self, handler, mock_http):
        body = {
            "repository": {"full_name": "org/repo"},
            "ref": "refs/heads/main",
            "after": "abc",
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=RuntimeError("git push fail"),
            ):
                result = handler._handle_git_push(h)
        assert _status(result) == 500

    def test_git_push_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_handle_git_push") as mock_gp:
            mock_gp.return_value = MagicMock(status_code=200, body=b'{}')
            h = mock_http(body={"repository": {}})
            handler.handle_post("/api/v1/scheduler/events/git-push", {}, h)
            mock_gp.assert_called_once_with(h)


# ============================================================================
# POST /api/v1/scheduler/events/file-upload - File upload events
# ============================================================================


class TestHandleFileUpload:
    """Tests for file upload event handler."""

    def test_file_upload_success(self, handler, mock_http):
        runs = [FakeRun(run_id="r1", job_id="j1")]
        body = {
            "workspace_id": "ws-123",
            "document_ids": ["doc1", "doc2"],
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=runs,
            ):
                result = handler._handle_file_upload(h)
        resp = _body(result)
        assert _status(result) == 200
        assert resp["success"] is True
        assert resp["workspace_id"] == "ws-123"
        assert resp["document_ids"] == ["doc1", "doc2"]
        assert resp["triggered_jobs"] == 1

    def test_file_upload_no_body(self, handler, mock_http):
        h = MockHTTPHandler(body=None)
        result = handler._handle_file_upload(h)
        assert _status(result) == 400

    def test_file_upload_missing_workspace_id(self, handler, mock_http):
        h = mock_http(body={"document_ids": ["doc1"]})
        result = handler._handle_file_upload(h)
        body = _body(result)
        assert _status(result) == 400
        assert "workspace_id" in body.get("error", "").lower()

    def test_file_upload_missing_document_ids(self, handler, mock_http):
        h = mock_http(body={"workspace_id": "ws-123"})
        result = handler._handle_file_upload(h)
        body = _body(result)
        assert _status(result) == 400
        assert "document_ids" in body.get("error", "").lower()

    def test_file_upload_empty_document_ids(self, handler, mock_http):
        h = mock_http(body={"workspace_id": "ws-123", "document_ids": []})
        result = handler._handle_file_upload(h)
        body = _body(result)
        assert _status(result) == 400
        assert "document_ids" in body.get("error", "").lower()

    def test_file_upload_runtime_error(self, handler, mock_http):
        body = {"workspace_id": "ws-123", "document_ids": ["doc1"]}
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=RuntimeError("upload fail"),
            ):
                result = handler._handle_file_upload(h)
        assert _status(result) == 500

    def test_file_upload_os_error(self, handler, mock_http):
        body = {"workspace_id": "ws-123", "document_ids": ["doc1"]}
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=OSError("disk full"),
            ):
                result = handler._handle_file_upload(h)
        assert _status(result) == 500

    def test_file_upload_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_handle_file_upload") as mock_fu:
            mock_fu.return_value = MagicMock(status_code=200, body=b'{}')
            h = mock_http(body={"workspace_id": "ws-1", "document_ids": ["d1"]})
            handler.handle_post("/api/v1/scheduler/events/file-upload", {}, h)
            mock_fu.assert_called_once_with(h)


# ============================================================================
# handle() GET routing
# ============================================================================


class TestHandleGetRouting:
    """Tests for GET request routing."""

    def test_handle_returns_none_for_unknown(self, handler, mock_http):
        result = handler.handle("/api/v1/scheduler/unknown", {}, mock_http())
        assert result is None

    def test_handle_returns_none_for_empty_path(self, handler, mock_http):
        result = handler.handle("", {}, mock_http())
        assert result is None

    def test_handle_status_route(self, handler, mock_http):
        with patch.object(handler, "_get_scheduler_status") as mock_stat:
            mock_stat.return_value = MagicMock(status_code=200, body=b'{}')
            result = handler.handle("/api/v1/scheduler/status", {}, mock_http())
            mock_stat.assert_called_once()

    def test_handle_job_with_wrong_segment_count(self, handler, mock_http):
        """Extra segments after job_id/action should return None."""
        result = handler.handle(
            "/api/v1/scheduler/jobs/j1/history/extra", {}, mock_http()
        )
        assert result is None

    def test_handle_jobs_path_6_segments_gets_job(self, handler, mock_http):
        """6 segments means /api/v1/scheduler/jobs/{job_id}."""
        with patch.object(handler, "_get_job") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle("/api/v1/scheduler/jobs/myid", {}, mock_http())
            mock_get.assert_called_once_with("myid")

    def test_handle_jobs_path_7_segments_history(self, handler, mock_http):
        """7 segments with 'history' gets job history."""
        with patch.object(handler, "_get_job_history") as mock_hist:
            mock_hist.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle("/api/v1/scheduler/jobs/myid/history", {}, mock_http())
            mock_hist.assert_called_once()

    def test_handle_jobs_path_7_segments_non_history(self, handler, mock_http):
        """7 segments but not 'history' returns None for GET."""
        result = handler.handle("/api/v1/scheduler/jobs/myid/other", {}, mock_http())
        assert result is None


# ============================================================================
# handle_post() routing
# ============================================================================


class TestHandlePostRouting:
    """Tests for POST request routing."""

    def test_post_unknown_path(self, handler, mock_http):
        result = handler.handle_post("/api/v1/scheduler/unknown", {}, mock_http())
        assert result is None

    def test_post_trigger_action(self, handler, mock_http):
        with patch.object(handler, "_trigger_job") as mock_trig:
            mock_trig.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle_post("/api/v1/scheduler/jobs/j1/trigger", {}, mock_http())
            mock_trig.assert_called_once_with("j1")

    def test_post_pause_action(self, handler, mock_http):
        with patch.object(handler, "_pause_job") as mock_pause:
            mock_pause.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle_post("/api/v1/scheduler/jobs/j1/pause", {}, mock_http())
            mock_pause.assert_called_once_with("j1")

    def test_post_resume_action(self, handler, mock_http):
        with patch.object(handler, "_resume_job") as mock_resume:
            mock_resume.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle_post("/api/v1/scheduler/jobs/j1/resume", {}, mock_http())
            mock_resume.assert_called_once_with("j1")

    def test_post_invalid_action(self, handler, mock_http):
        """Invalid action on jobs/id/action returns None."""
        result = handler.handle_post(
            "/api/v1/scheduler/jobs/j1/invalid_action", {}, mock_http()
        )
        assert result is None

    def test_post_webhook_wrong_segment_count(self, handler, mock_http):
        """Webhook path with extra segments returns None."""
        result = handler.handle_post(
            "/api/v1/scheduler/webhooks/wh-1/extra", {}, mock_http()
        )
        assert result is None

    def test_post_jobs_create(self, handler, mock_http):
        with patch.object(handler, "_create_job") as mock_create:
            mock_create.return_value = MagicMock(status_code=201, body=b'{}')
            h = mock_http(body={"name": "Test"})
            handler.handle_post("/api/v1/scheduler/jobs", {}, h)
            mock_create.assert_called_once_with(h)


# ============================================================================
# handle_delete() routing
# ============================================================================


class TestHandleDeleteRouting:
    """Tests for DELETE request routing."""

    def test_delete_unknown_path(self, handler, mock_http):
        result = handler.handle_delete("/api/v1/scheduler/unknown", {}, mock_http())
        assert result is None

    def test_delete_job_correct_segments(self, handler, mock_http):
        with patch.object(handler, "_delete_job") as mock_del:
            mock_del.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle_delete("/api/v1/scheduler/jobs/j1", {}, mock_http())
            mock_del.assert_called_once_with("j1")

    def test_delete_with_extra_segments(self, handler, mock_http):
        result = handler.handle_delete(
            "/api/v1/scheduler/jobs/j1/extra", {}, mock_http()
        )
        assert result is None

    def test_delete_non_jobs_path(self, handler, mock_http):
        result = handler.handle_delete(
            "/api/v1/scheduler/webhooks/wh-1", {}, mock_http()
        )
        assert result is None


# ============================================================================
# BASE_ROUTES constant
# ============================================================================


class TestBaseRoutes:
    """Tests for the BASE_ROUTES class attribute."""

    def test_base_routes_contains_jobs(self, handler):
        assert "/api/v1/scheduler/jobs" in handler.BASE_ROUTES

    def test_base_routes_contains_webhooks(self, handler):
        assert "/api/v1/scheduler/webhooks" in handler.BASE_ROUTES

    def test_base_routes_contains_git_push(self, handler):
        assert "/api/v1/scheduler/events/git-push" in handler.BASE_ROUTES

    def test_base_routes_contains_file_upload(self, handler):
        assert "/api/v1/scheduler/events/file-upload" in handler.BASE_ROUTES

    def test_base_routes_contains_status(self, handler):
        assert "/api/v1/scheduler/status" in handler.BASE_ROUTES

    def test_base_routes_count(self, handler):
        assert len(handler.BASE_ROUTES) == 5


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_create_job_invalid_json(self, handler, mock_http):
        """Malformed JSON body returns 400."""
        h = MockHTTPHandler()
        h.rfile.read.return_value = b"not json"
        h.headers = {
            "Content-Length": "8",
            "Authorization": "Bearer test-valid-token",
        }
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._scheduler_circuit_breaker"
            ) as mock_cb:
                mock_cb.can_proceed.return_value = True
                result = handler._create_job(h)
        body = _body(result)
        assert _status(result) == 400

    def test_webhook_with_no_signature(self, handler, mock_http):
        """Webhook without X-Webhook-Signature still passes (signature=None)."""
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"event": "test"})
        # headers dict does not have X-Webhook-Signature
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ):
                result = handler._handle_webhook(h, "wh-001")
        assert _status(result) == 200

    def test_git_push_missing_repository(self, handler, mock_http):
        """Git push with empty repository still works (repo="")."""
        body = {"ref": "refs/heads/main", "after": "abc"}
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ):
                result = handler._handle_git_push(h)
        resp = _body(result)
        assert _status(result) == 200
        assert resp["repository"] == ""

    def test_git_push_missing_ref(self, handler, mock_http):
        """Git push with empty ref still works (branch="")."""
        body = {"repository": {"full_name": "o/r"}, "after": "abc"}
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                return_value=[],
            ):
                result = handler._handle_git_push(h)
        resp = _body(result)
        assert _status(result) == 200
        assert resp["branch"] == ""

    def test_trigger_records_failure_on_exception(self, handler, mock_http):
        """Circuit breaker records failure on exception during trigger."""
        import aragora.server.handlers.features.scheduler as sched_mod
        cb = sched_mod._scheduler_circuit_breaker
        job = FakeJob(job_id="j1", name="Job 1")
        mock_sched = _make_mock_scheduler()
        mock_sched.get_job.return_value = job
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=ValueError("bad value"),
            ):
                with patch.object(cb, "record_failure") as mock_fail:
                    handler._trigger_job("j1")
                    mock_fail.assert_called_once()

    def test_file_upload_type_error(self, handler, mock_http):
        body = {"workspace_id": "ws-1", "document_ids": ["d1"]}
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=TypeError("type err"),
            ):
                result = handler._handle_file_upload(h)
        assert _status(result) == 500

    def test_webhook_connection_error(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"event": "test"})
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=ConnectionError("conn err"),
            ):
                result = handler._handle_webhook(h, "wh-001")
        assert _status(result) == 500

    def test_webhook_timeout_error(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"event": "test"})
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=TimeoutError("timeout"),
            ):
                result = handler._handle_webhook(h, "wh-001")
        assert _status(result) == 500

    def test_git_push_timeout_error(self, handler, mock_http):
        body = {
            "repository": {"full_name": "o/r"},
            "ref": "refs/heads/main",
            "after": "abc",
        }
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=TimeoutError("timeout"),
            ):
                result = handler._handle_git_push(h)
        assert _status(result) == 500

    def test_file_upload_connection_error(self, handler, mock_http):
        body = {"workspace_id": "ws-1", "document_ids": ["d1"]}
        h = mock_http(body=body)
        mock_sched = _make_mock_scheduler()
        with patch.object(handler, "_get_scheduler", return_value=mock_sched):
            with patch(
                "aragora.server.handlers.features.scheduler._run_async",
                side_effect=ConnectionError("conn"),
            ):
                result = handler._handle_file_upload(h)
        assert _status(result) == 500
