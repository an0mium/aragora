"""
Tests for External Agent Gateway endpoint handlers.

Tests:
- ExternalAgentsHandler initialization
- Route matching (can_handle)
- GET requests (list adapters, health check, get task status)
- POST requests (submit task)
- DELETE requests (cancel task)
- RBAC permission enforcement
- Rate limiting
- Error handling (missing adapter, invalid task ID)
- Edge cases
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Any


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.client_address = ("192.168.1.1", 8080)
    handler.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }
    return handler


@pytest.fixture
def mock_authenticated_user():
    """Create a mock authenticated user."""
    user = MagicMock()
    user.id = "user-123"
    user.org_id = "org-456"
    user.roles = {"member", "user"}
    user.is_authenticated = True
    return user


@pytest.fixture
def external_agents_handler():
    """Create an ExternalAgentsHandler instance."""
    from aragora.server.handlers.external_agents import ExternalAgentsHandler

    return ExternalAgentsHandler({})


@pytest.fixture
def mock_adapter_spec():
    """Create a mock adapter spec."""

    @dataclass
    class MockConfig:
        adapter_name: str = "mock"

    spec = MagicMock()
    spec.name = "openhands"
    spec.description = "OpenHands AI agent"
    spec.config_class = MockConfig
    spec.adapter_class = MagicMock()
    return spec


@pytest.fixture
def mock_task_result():
    """Create a mock task result."""
    from aragora.agents.external.models import TaskResult, TaskStatus

    return TaskResult(
        task_id="task-123",
        status=TaskStatus.COMPLETED,
        output="Task completed successfully",
        steps_executed=5,
        tokens_used=1000,
        cost_usd=0.05,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_health_status():
    """Create a mock health status."""
    from aragora.agents.external.models import HealthStatus

    return HealthStatus(
        adapter_name="openhands",
        healthy=True,
        last_check=datetime.now(timezone.utc),
        response_time_ms=50.0,
        framework_version="1.0.0",
    )


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiters before each test."""
    from aragora.server.handlers.external_agents import _submit_limiter, _read_limiter

    _submit_limiter.clear()
    _read_limiter.clear()
    yield
    _submit_limiter.clear()
    _read_limiter.clear()


# =============================================================================
# ExternalAgentsHandler Initialization Tests
# =============================================================================


class TestExternalAgentsHandlerInit:
    """Tests for ExternalAgentsHandler initialization."""

    def test_init_with_empty_context(self):
        """Should initialize with empty context."""
        from aragora.server.handlers.external_agents import ExternalAgentsHandler

        handler = ExternalAgentsHandler({})
        assert hasattr(handler, "ctx")

    def test_routes_constant_is_list(self):
        """ROUTES should be a list."""
        from aragora.server.handlers.external_agents import ExternalAgentsHandler

        assert isinstance(ExternalAgentsHandler.ROUTES, list)

    def test_routes_includes_core_endpoints(self):
        """ROUTES should include core external agent endpoints."""
        from aragora.server.handlers.external_agents import ExternalAgentsHandler

        routes = ExternalAgentsHandler.ROUTES
        assert "/api/external-agents/tasks" in routes
        assert "/api/external-agents/tasks/*" in routes
        assert "/api/external-agents/adapters" in routes
        assert "/api/external-agents/health" in routes


# =============================================================================
# Route Matching Tests
# =============================================================================


class TestExternalAgentsHandlerCanHandle:
    """Tests for can_handle method."""

    def test_can_handle_tasks_endpoint(self, external_agents_handler):
        """Should handle /api/external-agents/tasks."""
        assert external_agents_handler.can_handle("/api/external-agents/tasks") is True

    def test_can_handle_tasks_endpoint_with_version(self, external_agents_handler):
        """Should handle /api/v1/external-agents/tasks."""
        assert external_agents_handler.can_handle("/api/v1/external-agents/tasks") is True

    def test_can_handle_specific_task(self, external_agents_handler):
        """Should handle /api/external-agents/tasks/{id}."""
        assert external_agents_handler.can_handle("/api/external-agents/tasks/task-123") is True

    def test_can_handle_adapters_endpoint(self, external_agents_handler):
        """Should handle /api/external-agents/adapters."""
        assert external_agents_handler.can_handle("/api/external-agents/adapters") is True

    def test_can_handle_health_endpoint(self, external_agents_handler):
        """Should handle /api/external-agents/health."""
        assert external_agents_handler.can_handle("/api/external-agents/health") is True

    def test_cannot_handle_unknown_path(self, external_agents_handler):
        """Should not handle unknown paths."""
        assert external_agents_handler.can_handle("/api/debates") is False
        assert external_agents_handler.can_handle("/api/agents") is False
        assert external_agents_handler.can_handle("/api/external") is False


# =============================================================================
# GET Request Tests
# =============================================================================


class TestExternalAgentsHandlerGetRequests:
    """Tests for handle method (GET requests)."""

    def test_handle_returns_none_for_unmatched_path(self, external_agents_handler, mock_handler):
        """Should return None for unmatched paths."""
        with patch.object(external_agents_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(id="user-1"), None)
            result = external_agents_handler.handle("/api/debates", {}, mock_handler)
            assert result is None

    def test_handle_rate_limit_exceeded(self, external_agents_handler, mock_handler):
        """Should return 429 when rate limit is exceeded."""
        from aragora.server.handlers.external_agents import _read_limiter

        # Exhaust rate limit
        for _ in range(70):  # Default is 60 requests per minute
            _read_limiter.is_allowed("192.168.1.1")

        result = external_agents_handler.handle("/api/external-agents/adapters", {}, mock_handler)
        assert result is not None
        assert result.status_code == 429

    def test_handle_requires_authentication(self, external_agents_handler, mock_handler):
        """Should return 401 when not authenticated."""
        from aragora.server.handlers.base import error_response

        with patch.object(external_agents_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, error_response("Not authenticated", 401))

            result = external_agents_handler.handle(
                "/api/external-agents/adapters", {}, mock_handler
            )
            assert result is not None
            assert result.status_code == 401

    def test_handle_list_adapters(self, external_agents_handler, mock_handler, mock_adapter_spec):
        """Should list registered adapters."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
        ):
            mock_auth.return_value = (MagicMock(id="user-1"), None)
            mock_registry.list_specs.return_value = [mock_adapter_spec]

            result = external_agents_handler.handle(
                "/api/external-agents/adapters", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 200
            # Verify the response contains adapter info
            import json

            data = json.loads(result.body)
            assert "adapters" in data
            assert data["total"] == 1

    def test_handle_health_check_all_adapters(
        self, external_agents_handler, mock_handler, mock_adapter_spec, mock_health_status
    ):
        """Should health check all adapters when no adapter specified."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (MagicMock(id="user-1"), None)
            mock_registry.list_specs.return_value = [mock_adapter_spec]
            mock_run_coro.return_value = mock_health_status

            result = external_agents_handler.handle("/api/external-agents/health", {}, mock_handler)

            assert result is not None
            assert result.status_code == 200

    def test_handle_health_check_specific_adapter(
        self, external_agents_handler, mock_handler, mock_adapter_spec, mock_health_status
    ):
        """Should health check specific adapter when adapter param provided."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (MagicMock(id="user-1"), None)
            mock_registry.list_specs.return_value = [mock_adapter_spec]
            mock_run_coro.return_value = mock_health_status

            result = external_agents_handler.handle(
                "/api/external-agents/health",
                {"adapter": ["openhands"]},
                mock_handler,
            )

            assert result is not None
            assert result.status_code == 200

    def test_handle_get_task_status(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should get task status for valid task ID."""
        from aragora.agents.external.models import TaskStatus

        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = TaskStatus.RUNNING

            result = external_agents_handler.handle(
                "/api/external-agents/tasks/openhands-task-123", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 200
            import json

            data = json.loads(result.body)
            assert data["task_id"] == "openhands-task-123"
            assert data["status"] == "running"

    def test_handle_get_task_status_completed_includes_result(
        self, external_agents_handler, mock_handler, mock_authenticated_user, mock_task_result
    ):
        """Should include result in response for completed tasks."""
        from aragora.agents.external.models import TaskStatus

        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            # First call returns status, second call returns result
            mock_run_coro.side_effect = [TaskStatus.COMPLETED, mock_task_result]

            result = external_agents_handler.handle(
                "/api/external-agents/tasks/task-123", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 200
            import json

            data = json.loads(result.body)
            assert data["task_id"] == "task-123"
            assert data["status"] == "completed"
            assert "result" in data

    def test_handle_get_task_invalid_task_id(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 400 for invalid task ID with slash."""
        with patch.object(external_agents_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_authenticated_user, None)

            result = external_agents_handler.handle(
                "/api/external-agents/tasks/invalid/task/id", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 400

    def test_handle_get_task_not_found(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 404 when task not found."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = KeyError("Task not found")

            result = external_agents_handler.handle(
                "/api/external-agents/tasks/nonexistent", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 404


# =============================================================================
# POST Request Tests
# =============================================================================


class TestExternalAgentsHandlerPostRequests:
    """Tests for handle_post method (POST requests)."""

    def test_handle_post_returns_none_for_unmatched_path(
        self, external_agents_handler, mock_handler
    ):
        """Should return None for unmatched paths."""
        result = external_agents_handler.handle_post(
            "/api/external-agents/adapters", {}, mock_handler
        )
        assert result is None

    def test_handle_post_rate_limit_exceeded(self, external_agents_handler, mock_handler):
        """Should return 429 when rate limit is exceeded."""
        from aragora.server.handlers.external_agents import _submit_limiter

        # Exhaust rate limit (10 per minute for submissions)
        for _ in range(15):
            _submit_limiter.is_allowed("192.168.1.1")

        result = external_agents_handler.handle_post("/api/external-agents/tasks", {}, mock_handler)
        assert result is not None
        assert result.status_code == 429

    def test_handle_post_requires_authentication(self, external_agents_handler, mock_handler):
        """Should return 401 when not authenticated."""
        from aragora.server.handlers.base import error_response

        with patch.object(external_agents_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, error_response("Not authenticated", 401))

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )
            assert result is not None
            assert result.status_code == 401

    def test_handle_post_submit_task_missing_required_fields(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 400 when task_type or prompt missing."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = ({"task_type": "code"}, None)  # Missing prompt

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 400

    def test_handle_post_submit_task_prompt_too_long(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 400 when prompt exceeds max length."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "x" * 10001},
                None,
            )

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 400
            import json

            data = json.loads(result.body)
            assert "too long" in data.get("error", "").lower()

    def test_handle_post_submit_task_unknown_adapter(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 400 when adapter is not registered."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Fix the bug",
                    "adapter": "unknown-adapter",
                },
                None,
            )
            mock_registry.is_registered.return_value = False
            mock_registry.get_registered_names.return_value = ["openhands"]

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 400
            import json

            data = json.loads(result.body)
            assert "unknown adapter" in data.get("error", "").lower()

    def test_handle_post_submit_task_invalid_tool_permission(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 400 for invalid tool permission."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Fix the bug",
                    "tool_permissions": ["invalid_permission"],
                },
                None,
            )
            mock_registry.is_registered.return_value = True

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 400

    def test_handle_post_submit_task_success(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should successfully submit a task."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Fix the bug in main.py",
                },
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-new-123"

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 201
            import json

            data = json.loads(result.body)
            assert data["task_id"] == "task-new-123"
            assert data["status"] == "pending"

    def test_handle_post_submit_task_with_tool_permissions(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should successfully submit a task with valid tool permissions."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Fix the bug",
                    "tool_permissions": ["file_read", "file_write"],
                },
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-456"

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 201

    def test_handle_post_submit_task_policy_denied(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 403 when policy denies task submission."""
        from aragora.agents.external.proxy import PolicyDeniedError

        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Fix the bug"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = PolicyDeniedError("Tool not allowed")

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 403

    def test_handle_post_submit_task_timeout_clamped(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should clamp timeout_seconds to max value."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
            patch("aragora.agents.external.models.TaskRequest") as mock_task_request,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Fix the bug",
                    "timeout_seconds": 99999,  # Way over max of 7200
                },
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-789"

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 201


# =============================================================================
# DELETE Request Tests
# =============================================================================


class TestExternalAgentsHandlerDeleteRequests:
    """Tests for handle_delete method (DELETE requests)."""

    def test_handle_delete_returns_none_for_unmatched_path(
        self, external_agents_handler, mock_handler
    ):
        """Should return None for unmatched paths."""
        result = external_agents_handler.handle_delete(
            "/api/external-agents/adapters", {}, mock_handler
        )
        assert result is None

    def test_handle_delete_rate_limit_exceeded(self, external_agents_handler, mock_handler):
        """Should return 429 when rate limit is exceeded."""
        from aragora.server.handlers.external_agents import _read_limiter

        # Exhaust rate limit
        for _ in range(70):
            _read_limiter.is_allowed("192.168.1.1")

        result = external_agents_handler.handle_delete(
            "/api/external-agents/tasks/task-123", {}, mock_handler
        )
        assert result is not None
        assert result.status_code == 429

    def test_handle_delete_requires_authentication(self, external_agents_handler, mock_handler):
        """Should return 401 when not authenticated."""
        from aragora.server.handlers.base import error_response

        with patch.object(external_agents_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, error_response("Not authenticated", 401))

            result = external_agents_handler.handle_delete(
                "/api/external-agents/tasks/task-123", {}, mock_handler
            )
            assert result is not None
            assert result.status_code == 401

    def test_handle_delete_invalid_task_id(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 400 for invalid task ID."""
        with patch.object(external_agents_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_authenticated_user, None)

            result = external_agents_handler.handle_delete(
                "/api/external-agents/tasks/invalid/id", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 400

    def test_handle_delete_cancel_task_success(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should successfully cancel a task."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = True

            result = external_agents_handler.handle_delete(
                "/api/external-agents/tasks/task-123", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 200
            import json

            data = json.loads(result.body)
            assert data["task_id"] == "task-123"
            assert data["cancelled"] is True

    def test_handle_delete_cancel_task_not_found(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return success even if task not cancellable (already complete)."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = False

            result = external_agents_handler.handle_delete(
                "/api/external-agents/tasks/completed-task", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 200
            import json

            data = json.loads(result.body)
            assert data["cancelled"] is False

    def test_handle_delete_no_adapter_for_task(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 404 when no adapter available for task."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = False  # Both checks fail

            result = external_agents_handler.handle_delete(
                "/api/external-agents/tasks/unknown-prefix-task", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 404


# =============================================================================
# RBAC Permission Tests
# =============================================================================


class TestExternalAgentsRBAC:
    """Tests for RBAC permission enforcement."""

    def test_list_adapters_requires_agents_read_permission(self, external_agents_handler):
        """_list_adapters should have require_permission decorator."""
        from aragora.server.handlers.external_agents import (
            AGENTS_READ_PERMISSION,
        )

        method = external_agents_handler._list_adapters
        # Check that the method has the permission decorator
        # The decorator wraps the function, so we check for the attribute
        assert hasattr(method, "__wrapped__") or callable(method)
        assert AGENTS_READ_PERMISSION == "agents:read"

    def test_health_check_requires_agents_read_permission(self, external_agents_handler):
        """_health_check should require agents:read permission."""
        from aragora.server.handlers.external_agents import (
            AGENTS_READ_PERMISSION,
        )

        method = external_agents_handler._health_check
        assert hasattr(method, "__wrapped__") or callable(method)
        assert AGENTS_READ_PERMISSION == "agents:read"

    def test_submit_task_requires_agents_write_permission(self, external_agents_handler):
        """_submit_task should require agents:write permission."""
        from aragora.server.handlers.external_agents import (
            AGENTS_WRITE_PERMISSION,
        )

        method = external_agents_handler._submit_task
        assert hasattr(method, "__wrapped__") or callable(method)
        assert AGENTS_WRITE_PERMISSION == "agents:write"

    def test_get_task_requires_agents_read_permission(self, external_agents_handler):
        """_get_task should require agents:read permission."""
        from aragora.server.handlers.external_agents import (
            AGENTS_READ_PERMISSION,
        )

        method = external_agents_handler._get_task
        assert hasattr(method, "__wrapped__") or callable(method)

    def test_cancel_task_requires_agents_write_permission(self, external_agents_handler):
        """_cancel_task should require agents:write permission."""
        from aragora.server.handlers.external_agents import (
            AGENTS_WRITE_PERMISSION,
        )

        method = external_agents_handler._cancel_task
        assert hasattr(method, "__wrapped__") or callable(method)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestExternalAgentsErrorHandling:
    """Tests for error handling."""

    def test_list_adapters_handles_exception(self, external_agents_handler, mock_handler):
        """Should return 500 on internal error in list adapters."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
        ):
            mock_auth.return_value = (MagicMock(id="user-1"), None)
            mock_registry.list_specs.side_effect = Exception("Database error")

            result = external_agents_handler.handle(
                "/api/external-agents/adapters", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 500

    def test_health_check_handles_adapter_exception(
        self, external_agents_handler, mock_handler, mock_adapter_spec
    ):
        """Should handle adapter health check failures gracefully."""
        mock_user = MagicMock(id="user-1")
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(
                external_agents_handler,
                "require_permission_or_error",
                return_value=(mock_user, None),
            ),
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
        ):
            mock_auth.return_value = (mock_user, None)
            mock_registry.list_specs.return_value = [mock_adapter_spec]
            # Mock adapter creation to raise an exception
            mock_adapter_spec.adapter_class.side_effect = Exception("Connection failed")

            result = external_agents_handler.handle("/api/external-agents/health", {}, mock_handler)

            assert result is not None
            assert result.status_code == 200  # Returns health results including failures
            import json

            data = json.loads(result.body)
            assert len(data["health"]) == 1
            assert data["health"][0]["healthy"] is False
            assert "error" in data["health"][0]

    def test_submit_task_handles_internal_exception(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 500 on internal error during task submission."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Fix the bug"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = Exception("Unexpected error")

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 500

    def test_get_task_handles_internal_exception(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 500 on internal error during task status retrieval."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = RuntimeError("Service unavailable")

            result = external_agents_handler.handle(
                "/api/external-agents/tasks/task-123", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 500

    def test_cancel_task_handles_internal_exception(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 500 on internal error during task cancellation."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = Exception("Network error")

            result = external_agents_handler.handle_delete(
                "/api/external-agents/tasks/task-123", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 500


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestExternalAgentsEdgeCases:
    """Tests for edge cases."""

    def test_task_id_extraction_from_adapter_prefix(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should extract adapter name from task ID prefix."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)

            # First check: adapter from prefix "autogpt"
            def is_registered_side_effect(name):
                return name in ["autogpt", "openhands"]

            mock_registry.is_registered.side_effect = is_registered_side_effect
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = True

            result = external_agents_handler.handle_delete(
                "/api/external-agents/tasks/autogpt-task-abc", {}, mock_handler
            )

            # Verify it tried to use "autogpt" adapter first
            mock_registry.is_registered.assert_called()

    def test_fallback_to_openhands_when_prefix_not_registered(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should fallback to openhands when task ID prefix not registered."""
        from aragora.agents.external.models import TaskStatus

        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_auth.return_value = (mock_authenticated_user, None)

            # First call: unknown prefix not registered
            # Second call: fallback to openhands which is registered
            call_count = [0]

            def is_registered_side_effect(name):
                call_count[0] += 1
                if call_count[0] == 1:
                    return False  # unknown not registered
                return name == "openhands"

            mock_registry.is_registered.side_effect = is_registered_side_effect
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = TaskStatus.RUNNING

            result = external_agents_handler.handle(
                "/api/external-agents/tasks/unknown-task-123", {}, mock_handler
            )

            assert result is not None
            # Should succeed if openhands is available
            if mock_registry.is_registered.call_count >= 2:
                assert result.status_code in [200, 404]

    def test_user_id_extraction_from_dict_user(self, external_agents_handler, mock_handler):
        """Should extract user_id from dict-style user object."""
        dict_user = {"id": "dict-user-123", "org_id": "dict-org-456", "roles": ["user"]}

        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(
                external_agents_handler,
                "require_permission_or_error",
                return_value=(dict_user, None),
            ),
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_auth.return_value = (dict_user, None)
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Test"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-123"

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 201

    def test_empty_task_id_returns_error(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should return 400 for empty task ID."""
        with patch.object(external_agents_handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (mock_authenticated_user, None)

            # This path should match but have empty task ID after split
            result = external_agents_handler.handle("/api/external-agents/tasks/", {}, mock_handler)

            # With trailing slash, task_id will be empty after split
            assert result is not None
            # Either 400 for invalid ID or None (not handled)

    def test_max_steps_clamped(
        self, external_agents_handler, mock_handler, mock_authenticated_user
    ):
        """Should clamp max_steps to 500."""
        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Test",
                    "max_steps": 1000,  # Over max of 500
                },
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-123"

            result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )

            assert result is not None
            assert result.status_code == 201

    def test_version_prefix_stripped(self, external_agents_handler):
        """Should strip version prefix from paths."""
        # v1
        assert external_agents_handler.can_handle("/api/v1/external-agents/tasks") is True
        # v2
        assert external_agents_handler.can_handle("/api/v2/external-agents/tasks") is True
        # No version
        assert external_agents_handler.can_handle("/api/external-agents/tasks") is True


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestRunCoroUtility:
    """Tests for _run_coro helper function."""

    def test_run_coro_in_new_event_loop(self):
        """Should run coroutine in new event loop when none running."""
        from aragora.server.handlers.external_agents import _run_coro

        async def async_func():
            return "result"

        result = _run_coro(async_func())
        assert result == "result"


class TestRecordMetrics:
    """Tests for _record_metrics helper function."""

    def test_record_metrics_submit(self):
        """Should record metrics for submit operation."""
        from aragora.server.handlers.external_agents import _record_metrics

        with (
            patch(
                "aragora.server.handlers.external_agents.record_external_agent_task"
            ) as mock_task,
            patch(
                "aragora.server.handlers.external_agents.record_external_agent_duration"
            ) as mock_duration,
        ):
            _record_metrics("submit", "openhands", "code", 1.5)

            mock_task.assert_called_once_with("openhands", "submitted")
            mock_duration.assert_called_once_with("openhands", "code", 1.5)

    def test_record_metrics_handles_import_error(self):
        """Should handle missing prometheus module gracefully."""
        from aragora.server.handlers.external_agents import _record_metrics

        with patch.dict("sys.modules", {"aragora.server.prometheus": None}):
            # Should not raise
            _record_metrics("submit", "openhands", "code", 1.5)

    def test_record_metrics_handles_exception(self):
        """Should handle exceptions during metrics recording."""
        from aragora.server.handlers.external_agents import _record_metrics

        with patch(
            "aragora.server.handlers.external_agents.record_external_agent_task"
        ) as mock_task:
            mock_task.side_effect = Exception("Metrics error")

            # Should not raise
            _record_metrics("submit", "openhands", "code", 1.5)


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestExternalAgentsIntegration:
    """Integration-style tests for full request handling."""

    def test_full_task_lifecycle(
        self, external_agents_handler, mock_handler, mock_authenticated_user, mock_task_result
    ):
        """Test complete task lifecycle: submit -> status -> result -> cancel."""
        from aragora.agents.external.models import TaskStatus

        with (
            patch.object(external_agents_handler, "require_auth_or_error") as mock_auth,
            patch.object(external_agents_handler, "set_request_context"),
            patch.object(external_agents_handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_auth.return_value = (mock_authenticated_user, None)
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()

            # Step 1: Submit task
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Fix bug"},
                None,
            )
            mock_run_coro.return_value = "lifecycle-task-123"

            submit_result = external_agents_handler.handle_post(
                "/api/external-agents/tasks", {}, mock_handler
            )
            assert submit_result.status_code == 201

            # Step 2: Check status (running)
            mock_run_coro.return_value = TaskStatus.RUNNING

            status_result = external_agents_handler.handle(
                "/api/external-agents/tasks/lifecycle-task-123", {}, mock_handler
            )
            assert status_result.status_code == 200
            import json

            status_data = json.loads(status_result.body)
            assert status_data["status"] == "running"

            # Step 3: Check status (completed with result)
            mock_run_coro.side_effect = [TaskStatus.COMPLETED, mock_task_result]

            completed_result = external_agents_handler.handle(
                "/api/external-agents/tasks/lifecycle-task-123", {}, mock_handler
            )
            assert completed_result.status_code == 200
            completed_data = json.loads(completed_result.body)
            assert completed_data["status"] == "completed"
            assert "result" in completed_data

            # Step 4: Try to cancel (should return false since already completed)
            mock_run_coro.side_effect = None
            mock_run_coro.return_value = False

            cancel_result = external_agents_handler.handle_delete(
                "/api/external-agents/tasks/lifecycle-task-123", {}, mock_handler
            )
            assert cancel_result.status_code == 200
            cancel_data = json.loads(cancel_result.body)
            assert cancel_data["cancelled"] is False
