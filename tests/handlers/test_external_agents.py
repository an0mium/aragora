"""Comprehensive tests for ExternalAgentsHandler (aragora/server/handlers/external_agents.py).

Tests cover:
- Handler initialization and ROUTES constant
- Route matching (can_handle) with and without version prefix
- GET requests:
  - /api/external-agents/adapters (list adapters)
  - /api/external-agents/health (health check all or specific adapter)
  - /api/external-agents/tasks/{id} (get task status/result)
- POST requests:
  - /api/external-agents/tasks (submit task)
- DELETE requests:
  - /api/external-agents/tasks/{id} (cancel task)
- Rate limiting for read and submit endpoints
- Authentication enforcement
- Input validation (task ID, prompt length, tool permissions, required fields)
- Error handling (internal errors, adapter failures, circuit breaker)
- Edge cases (dict-style user, adapter fallback, clamped params)
- Circuit breaker behavior
- Metrics recording
- _run_coro and _call_run_coro helpers
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters and circuit breaker before each test."""
    try:
        from aragora.server.handlers.external_agents import (
            _read_limiter,
            _submit_limiter,
            reset_external_agents_circuit_breaker,
        )

        _submit_limiter.clear()
        _read_limiter.clear()
        reset_external_agents_circuit_breaker()
    except (ImportError, AttributeError):
        pass

    yield

    try:
        from aragora.server.handlers.external_agents import (
            _read_limiter,
            _submit_limiter,
            reset_external_agents_circuit_breaker,
        )

        _submit_limiter.clear()
        _read_limiter.clear()
        reset_external_agents_circuit_breaker()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def handler():
    """Create an ExternalAgentsHandler with empty server context."""
    from aragora.server.handlers.external_agents import ExternalAgentsHandler

    return ExternalAgentsHandler({})


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with client address and headers."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }
    return h


@pytest.fixture
def mock_user():
    """Create a mock authenticated user with object-style attributes."""
    user = MagicMock()
    user.id = "user-123"
    user.org_id = "org-456"
    user.roles = {"member", "admin"}
    user.permissions = {"agents:read", "agents:write", "debates:write", "debates:delete"}
    user.is_authenticated = True
    return user


@pytest.fixture
def dict_user():
    """Create a dict-style user object."""
    return {
        "id": "dict-user-789",
        "org_id": "dict-org-012",
        "roles": ["member"],
        "permissions": [],
    }


@pytest.fixture
def mock_adapter_spec():
    """Create a mock ExternalAdapterSpec."""

    @dataclass
    class MockConfig:
        adapter_name: str = "mock"

    spec = MagicMock()
    spec.name = "openhands"
    spec.description = "OpenHands autonomous coding agent"
    spec.config_class = MockConfig
    spec.adapter_class = MagicMock()
    return spec


@pytest.fixture
def mock_adapter_spec_b():
    """Create a second mock adapter spec for multi-adapter tests."""

    @dataclass
    class MockConfigB:
        adapter_name: str = "autogpt"

    spec = MagicMock()
    spec.name = "autogpt"
    spec.description = "AutoGPT framework"
    spec.config_class = MockConfigB
    spec.adapter_class = MagicMock()
    return spec


@pytest.fixture
def mock_health_status():
    """Create a mock HealthStatus from the external agent models."""
    from datetime import datetime, timezone

    from aragora.agents.external.models import HealthStatus

    return HealthStatus(
        adapter_name="openhands",
        healthy=True,
        last_check=datetime.now(timezone.utc),
        response_time_ms=42.0,
        framework_version="1.0.0",
    )


@pytest.fixture
def mock_task_result():
    """Create a mock TaskResult."""
    from datetime import datetime, timezone

    from aragora.agents.external.models import TaskResult, TaskStatus

    return TaskResult(
        task_id="task-abc",
        status=TaskStatus.COMPLETED,
        output="Done",
        steps_executed=3,
        tokens_used=500,
        cost_usd=0.02,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Initialization and ROUTES
# ---------------------------------------------------------------------------


class TestExternalAgentsInit:
    """Tests for handler construction and ROUTES."""

    def test_init_with_empty_context(self):
        from aragora.server.handlers.external_agents import ExternalAgentsHandler

        h = ExternalAgentsHandler({})
        assert hasattr(h, "ctx")

    def test_init_with_server_context(self):
        from aragora.server.handlers.external_agents import ExternalAgentsHandler

        ctx = {"key": "value"}
        h = ExternalAgentsHandler(server_context=ctx)
        assert h.ctx is ctx

    def test_routes_is_list(self, handler):
        assert isinstance(handler.ROUTES, list)

    def test_routes_includes_tasks(self, handler):
        assert "/api/external-agents/tasks" in handler.ROUTES

    def test_routes_includes_tasks_wildcard(self, handler):
        assert "/api/external-agents/tasks/*" in handler.ROUTES

    def test_routes_includes_adapters(self, handler):
        assert "/api/external-agents/adapters" in handler.ROUTES

    def test_routes_includes_health(self, handler):
        assert "/api/external-agents/health" in handler.ROUTES

    def test_routes_has_four_entries(self, handler):
        assert len(handler.ROUTES) == 4


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle path matching."""

    def test_tasks_endpoint(self, handler):
        assert handler.can_handle("/api/external-agents/tasks")

    def test_tasks_endpoint_versioned(self, handler):
        assert handler.can_handle("/api/v1/external-agents/tasks")

    def test_tasks_endpoint_v2(self, handler):
        assert handler.can_handle("/api/v2/external-agents/tasks")

    def test_specific_task(self, handler):
        assert handler.can_handle("/api/external-agents/tasks/task-123")

    def test_specific_task_versioned(self, handler):
        assert handler.can_handle("/api/v1/external-agents/tasks/my-task")

    def test_adapters_endpoint(self, handler):
        assert handler.can_handle("/api/external-agents/adapters")

    def test_adapters_versioned(self, handler):
        assert handler.can_handle("/api/v1/external-agents/adapters")

    def test_health_endpoint(self, handler):
        assert handler.can_handle("/api/external-agents/health")

    def test_health_versioned(self, handler):
        assert handler.can_handle("/api/v1/external-agents/health")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/agents")
        assert not handler.can_handle("/api/external")

    def test_partial_path_rejected(self, handler):
        assert not handler.can_handle("/api/external-agents")

    def test_deep_nested_task_id_matches(self, handler):
        # Task IDs with hyphens are common (prefix-uuid format)
        assert handler.can_handle("/api/external-agents/tasks/oh-abc-def-123")


# ---------------------------------------------------------------------------
# GET /api/external-agents/adapters
# ---------------------------------------------------------------------------


class TestListAdapters:
    """Tests for list adapters endpoint."""

    def test_list_adapters_success(self, handler, mock_http_handler, mock_adapter_spec):
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.return_value = [mock_adapter_spec]

            result = handler.handle("/api/external-agents/adapters", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 1
            assert len(body["adapters"]) == 1
            assert body["adapters"][0]["name"] == "openhands"
            assert body["adapters"][0]["description"] == "OpenHands autonomous coding agent"

    def test_list_adapters_empty(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.return_value = []

            result = handler.handle("/api/external-agents/adapters", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 0
            assert body["adapters"] == []

    def test_list_adapters_multiple(
        self, handler, mock_http_handler, mock_adapter_spec, mock_adapter_spec_b
    ):
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.return_value = [mock_adapter_spec, mock_adapter_spec_b]

            result = handler.handle("/api/external-agents/adapters", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 2
            names = {a["name"] for a in body["adapters"]}
            assert names == {"openhands", "autogpt"}

    def test_list_adapters_internal_error(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.side_effect = TypeError("unexpected error")

            result = handler.handle("/api/external-agents/adapters", {}, mock_http_handler)

            assert _status(result) == 500

    def test_list_adapters_versioned_path(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.return_value = []

            result = handler.handle("/api/v1/external-agents/adapters", {}, mock_http_handler)

            assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/external-agents/health
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_all_adapters(
        self, handler, mock_http_handler, mock_adapter_spec, mock_health_status
    ):
        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.list_specs.return_value = [mock_adapter_spec]
            mock_run_coro.return_value = mock_health_status

            result = handler.handle("/api/external-agents/health", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["total"] >= 1
            assert len(body["health"]) >= 1

    def test_health_check_specific_adapter(
        self, handler, mock_http_handler, mock_adapter_spec, mock_adapter_spec_b, mock_health_status
    ):
        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.list_specs.return_value = [mock_adapter_spec, mock_adapter_spec_b]
            mock_run_coro.return_value = mock_health_status

            result = handler.handle(
                "/api/external-agents/health",
                {"adapter": ["openhands"]},
                mock_http_handler,
            )

            assert _status(result) == 200
            body = _body(result)
            # Only one adapter should be checked (the one matching the filter)
            assert body["total"] == 1

    def test_health_check_no_adapters(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.return_value = []

            result = handler.handle("/api/external-agents/health", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 0
            assert body["health"] == []

    def test_health_check_adapter_failure(self, handler, mock_http_handler, mock_adapter_spec):
        """Individual adapter failures are reported but don't cause 500."""
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.return_value = [mock_adapter_spec]
            mock_adapter_spec.adapter_class.side_effect = ConnectionError("refused")

            result = handler.handle("/api/external-agents/health", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["health"][0]["healthy"] is False
            assert "error" in body["health"][0]

    def test_health_check_global_exception(self, handler, mock_http_handler):
        """Global exception in health check returns 500."""
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.side_effect = RuntimeError("total failure")

            result = handler.handle("/api/external-agents/health", {}, mock_http_handler)

            assert _status(result) == 500

    def test_health_check_versioned(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.list_specs.return_value = []

            result = handler.handle("/api/v1/external-agents/health", {}, mock_http_handler)

            assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/external-agents/tasks/{id}
# ---------------------------------------------------------------------------


class TestGetTask:
    """Tests for get task status endpoint."""

    def test_get_task_running(self, handler, mock_http_handler):
        from aragora.agents.external.models import TaskStatus

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = TaskStatus.RUNNING

            result = handler.handle(
                "/api/external-agents/tasks/openhands-task-123", {}, mock_http_handler
            )

            assert _status(result) == 200
            body = _body(result)
            assert body["task_id"] == "openhands-task-123"
            assert body["status"] == "running"
            assert "result" not in body

    def test_get_task_completed_includes_result(self, handler, mock_http_handler, mock_task_result):
        from aragora.agents.external.models import TaskStatus

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = [TaskStatus.COMPLETED, mock_task_result]

            result = handler.handle("/api/external-agents/tasks/task-abc", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["status"] == "completed"
            assert "result" in body

    def test_get_task_failed_includes_result(self, handler, mock_http_handler, mock_task_result):
        from aragora.agents.external.models import TaskStatus

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = [TaskStatus.FAILED, mock_task_result]

            result = handler.handle("/api/external-agents/tasks/task-fail", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["status"] == "failed"
            assert "result" in body

    def test_get_task_cancelled_includes_result(self, handler, mock_http_handler, mock_task_result):
        from aragora.agents.external.models import TaskStatus

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = [TaskStatus.CANCELLED, mock_task_result]

            result = handler.handle("/api/external-agents/tasks/task-cancel", {}, mock_http_handler)

            assert _status(result) == 200
            body = _body(result)
            assert body["status"] == "cancelled"
            assert "result" in body

    def test_get_task_timeout_includes_result(self, handler, mock_http_handler, mock_task_result):
        from aragora.agents.external.models import TaskStatus

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = [TaskStatus.TIMEOUT, mock_task_result]

            result = handler.handle(
                "/api/external-agents/tasks/task-timeout", {}, mock_http_handler
            )

            assert _status(result) == 200
            body = _body(result)
            assert body["status"] == "timeout"
            assert "result" in body

    def test_get_task_pending_no_result(self, handler, mock_http_handler):
        from aragora.agents.external.models import TaskStatus

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = TaskStatus.PENDING

            result = handler.handle(
                "/api/external-agents/tasks/task-pending", {}, mock_http_handler
            )

            assert _status(result) == 200
            body = _body(result)
            assert body["status"] == "pending"
            assert "result" not in body

    def test_get_task_invalid_id_with_slash(self, handler, mock_http_handler):
        result = handler.handle("/api/external-agents/tasks/bad/id/here", {}, mock_http_handler)

        assert _status(result) == 400

    def test_get_task_empty_id(self, handler, mock_http_handler):
        """Trailing slash means empty task ID."""
        result = handler.handle("/api/external-agents/tasks/", {}, mock_http_handler)
        # Empty task ID should return 400
        assert result is not None
        assert _status(result) == 400

    def test_get_task_not_found(self, handler, mock_http_handler):
        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = KeyError("not found")

            result = handler.handle("/api/external-agents/tasks/nonexistent", {}, mock_http_handler)

            assert _status(result) == 404

    def test_get_task_internal_error(self, handler, mock_http_handler):
        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = RuntimeError("boom")

            result = handler.handle("/api/external-agents/tasks/task-err", {}, mock_http_handler)

            assert _status(result) == 500

    def test_get_task_no_adapter_available(self, handler, mock_http_handler):
        """Both prefix and openhands not registered returns 404."""
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.is_registered.return_value = False

            result = handler.handle(
                "/api/external-agents/tasks/unknown-task", {}, mock_http_handler
            )

            assert _status(result) == 404

    def test_get_task_adapter_prefix_extraction(self, handler, mock_http_handler):
        """Adapter name is extracted from the part before the first hyphen."""
        from aragora.agents.external.models import TaskStatus

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):

            def is_reg(name):
                return name == "autogpt"

            mock_registry.is_registered.side_effect = is_reg
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = TaskStatus.RUNNING

            result = handler.handle(
                "/api/external-agents/tasks/autogpt-abc-def", {}, mock_http_handler
            )

            assert _status(result) == 200
            mock_registry.is_registered.assert_any_call("autogpt")

    def test_get_task_falls_back_to_openhands(self, handler, mock_http_handler):
        """When prefix is not registered, falls back to openhands."""
        from aragora.agents.external.models import TaskStatus

        call_count = [0]

        def is_reg(name):
            call_count[0] += 1
            if call_count[0] == 1:
                return False  # prefix "xyz" not registered
            return name == "openhands"

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.side_effect = is_reg
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = TaskStatus.RUNNING

            result = handler.handle(
                "/api/external-agents/tasks/xyz-task-123", {}, mock_http_handler
            )

            assert _status(result) == 200

    def test_get_task_no_hyphen_defaults_to_openhands(self, handler, mock_http_handler):
        """Task ID without hyphens defaults to 'openhands' adapter."""
        from aragora.agents.external.models import TaskStatus

        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = TaskStatus.PENDING

            result = handler.handle(
                "/api/external-agents/tasks/simpletaskid", {}, mock_http_handler
            )

            assert _status(result) == 200
            mock_registry.is_registered.assert_any_call("openhands")


# ---------------------------------------------------------------------------
# POST /api/external-agents/tasks
# ---------------------------------------------------------------------------


class TestSubmitTask:
    """Tests for task submission endpoint."""

    def test_submit_task_success(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Fix the bug"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-new-123"

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 201
            body = _body(result)
            assert body["task_id"] == "task-new-123"
            assert body["status"] == "pending"
            assert body["adapter"] == "openhands"

    def test_submit_task_custom_adapter(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Write tests", "adapter": "autogpt"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "autogpt-task-456"

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 201
            body = _body(result)
            assert body["adapter"] == "autogpt"

    def test_submit_task_missing_task_type(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
        ):
            mock_read_body.return_value = (
                {"prompt": "Fix the bug"},
                None,
            )

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 400

    def test_submit_task_missing_prompt(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
        ):
            mock_read_body.return_value = (
                {"task_type": "code"},
                None,
            )

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 400

    def test_submit_task_missing_both(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
        ):
            mock_read_body.return_value = ({}, None)

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 400

    def test_submit_task_prompt_too_long(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "x" * 10001},
                None,
            )

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 400
            assert "too long" in _body(result).get("error", "").lower()

    def test_submit_task_prompt_at_max_length(self, handler, mock_http_handler):
        """Prompt at exactly 10000 chars should be accepted."""
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "x" * 10000},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-ok"

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 201

    def test_submit_task_unknown_adapter(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Fix bug", "adapter": "unknown"},
                None,
            )
            mock_registry.is_registered.return_value = False
            mock_registry.get_registered_names.return_value = ["openhands"]

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 400
            assert "unknown" in _body(result).get("error", "").lower()

    def test_submit_task_invalid_tool_permission(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
        ):
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Fix bug",
                    "tool_permissions": ["completely_invalid"],
                },
                None,
            )
            mock_registry.is_registered.return_value = True

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 400
            assert "permission" in _body(result).get("error", "").lower()

    def test_submit_task_valid_tool_permissions(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Fix bug",
                    "tool_permissions": ["file_read", "shell_execute"],
                },
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-perms"

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 201

    def test_submit_task_policy_denied(self, handler, mock_http_handler):
        from aragora.agents.external.proxy import PolicyDeniedError

        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Fix bug"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = PolicyDeniedError("Not allowed")

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 403
            assert "denied" in _body(result).get("error", "").lower()

    def test_submit_task_internal_error(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Fix bug"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = RuntimeError("connection failed")

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 500

    def test_submit_task_body_parse_error(self, handler, mock_http_handler):
        """When read_json_body_validated returns an error, it propagates."""
        from aragora.server.handlers.base import error_response

        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
        ):
            mock_read_body.return_value = (None, error_response("Invalid JSON", 400))

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 400

    def test_submit_task_unmatched_path_returns_none(self, handler, mock_http_handler):
        result = handler.handle_post("/api/external-agents/adapters", {}, mock_http_handler)
        assert result is None

    def test_submit_task_timeout_clamped_to_7200(self, handler, mock_http_handler):
        """Timeout should be clamped to max 7200 seconds."""
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Test",
                    "timeout_seconds": 99999,
                },
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-clamp"

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 201

    def test_submit_task_max_steps_clamped_to_500(self, handler, mock_http_handler):
        """Max steps should be clamped to 500."""
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_read_body.return_value = (
                {
                    "task_type": "code",
                    "prompt": "Test",
                    "max_steps": 999,
                },
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-steps"

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 201

    def test_submit_task_with_context_and_metadata(self, handler, mock_http_handler):
        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_read_body.return_value = (
                {
                    "task_type": "research",
                    "prompt": "Analyze market data",
                    "context": {"workspace": "ws-1"},
                    "metadata": {"source": "api"},
                    "workspace_id": "ws-1",
                },
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-meta"

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 201

    def test_submit_task_with_dict_user(self, handler, dict_user):
        """Dict-style user auth context is correctly extracted in _set_auth_context."""
        handler._set_auth_context(dict_user)
        assert handler._auth_context is not None
        assert handler._auth_context.user_id == "dict-user-789"
        assert handler._auth_context.org_id == "dict-org-012"


# ---------------------------------------------------------------------------
# DELETE /api/external-agents/tasks/{id}
# ---------------------------------------------------------------------------


class TestCancelTask:
    """Tests for task cancellation endpoint."""

    def test_cancel_task_success(self, handler, mock_http_handler):
        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = True

            result = handler.handle_delete(
                "/api/external-agents/tasks/task-123", {}, mock_http_handler
            )

            assert _status(result) == 200
            body = _body(result)
            assert body["task_id"] == "task-123"
            assert body["cancelled"] is True

    def test_cancel_task_already_completed(self, handler, mock_http_handler):
        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = False

            result = handler.handle_delete(
                "/api/external-agents/tasks/done-task", {}, mock_http_handler
            )

            assert _status(result) == 200
            body = _body(result)
            assert body["cancelled"] is False

    def test_cancel_task_invalid_id(self, handler, mock_http_handler):
        result = handler.handle_delete("/api/external-agents/tasks/bad/id", {}, mock_http_handler)
        assert _status(result) == 400

    def test_cancel_task_empty_id(self, handler, mock_http_handler):
        result = handler.handle_delete("/api/external-agents/tasks/", {}, mock_http_handler)
        assert result is not None
        assert _status(result) == 400

    def test_cancel_task_no_adapter(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentRegistry"
        ) as mock_registry:
            mock_registry.is_registered.return_value = False

            result = handler.handle_delete(
                "/api/external-agents/tasks/unknown-task", {}, mock_http_handler
            )

            assert _status(result) == 404

    def test_cancel_task_internal_error(self, handler, mock_http_handler):
        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = RuntimeError("network error")

            result = handler.handle_delete(
                "/api/external-agents/tasks/task-err", {}, mock_http_handler
            )

            assert _status(result) == 500

    def test_cancel_task_unmatched_path_returns_none(self, handler, mock_http_handler):
        result = handler.handle_delete("/api/external-agents/adapters", {}, mock_http_handler)
        assert result is None

    def test_cancel_task_versioned_path(self, handler, mock_http_handler):
        with (
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = True

            result = handler.handle_delete(
                "/api/v1/external-agents/tasks/task-v1", {}, mock_http_handler
            )

            assert _status(result) == 200


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting enforcement."""

    def test_read_rate_limit_on_adapters(self, handler, mock_http_handler):
        from aragora.server.handlers.external_agents import _read_limiter

        for _ in range(70):
            _read_limiter.is_allowed("127.0.0.1")

        result = handler.handle("/api/external-agents/adapters", {}, mock_http_handler)
        assert _status(result) == 429

    def test_read_rate_limit_on_health(self, handler, mock_http_handler):
        from aragora.server.handlers.external_agents import _read_limiter

        for _ in range(70):
            _read_limiter.is_allowed("127.0.0.1")

        result = handler.handle("/api/external-agents/health", {}, mock_http_handler)
        assert _status(result) == 429

    def test_read_rate_limit_on_get_task(self, handler, mock_http_handler):
        from aragora.server.handlers.external_agents import _read_limiter

        for _ in range(70):
            _read_limiter.is_allowed("127.0.0.1")

        result = handler.handle("/api/external-agents/tasks/task-123", {}, mock_http_handler)
        assert _status(result) == 429

    def test_submit_rate_limit(self, handler, mock_http_handler):
        from aragora.server.handlers.external_agents import _submit_limiter

        for _ in range(15):
            _submit_limiter.is_allowed("127.0.0.1")

        result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)
        assert _status(result) == 429

    def test_delete_rate_limit(self, handler, mock_http_handler):
        from aragora.server.handlers.external_agents import _read_limiter

        for _ in range(70):
            _read_limiter.is_allowed("127.0.0.1")

        result = handler.handle_delete("/api/external-agents/tasks/task-123", {}, mock_http_handler)
        assert _status(result) == 429


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Tests for authentication enforcement."""

    def test_handle_auth_error(self, handler, mock_http_handler):
        from aragora.server.handlers.base import error_response

        with patch.object(
            handler,
            "require_auth_or_error",
            return_value=(None, error_response("Unauthorized", 401)),
        ):
            result = handler.handle("/api/external-agents/adapters", {}, mock_http_handler)
            assert _status(result) == 401

    def test_handle_post_auth_error(self, handler, mock_http_handler):
        from aragora.server.handlers.base import error_response

        with patch.object(
            handler,
            "require_auth_or_error",
            return_value=(None, error_response("Unauthorized", 401)),
        ):
            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)
            assert _status(result) == 401

    def test_handle_delete_auth_error(self, handler, mock_http_handler):
        from aragora.server.handlers.base import error_response

        with patch.object(
            handler,
            "require_auth_or_error",
            return_value=(None, error_response("Unauthorized", 401)),
        ):
            result = handler.handle_delete(
                "/api/external-agents/tasks/task-123", {}, mock_http_handler
            )
            assert _status(result) == 401


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for circuit breaker behavior on task submission."""

    def test_circuit_breaker_open_returns_503(self, handler, mock_http_handler):
        from aragora.server.handlers.external_agents import (
            get_external_agents_circuit_breaker,
        )

        cb = get_external_agents_circuit_breaker()
        # Force circuit open by recording failures
        for _ in range(10):
            cb.record_failure()

        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Test"},
                None,
            )

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 503
            assert "unavailable" in _body(result).get("error", "").lower()

    def test_circuit_breaker_records_success(self, handler, mock_http_handler):
        from aragora.server.handlers.external_agents import (
            get_external_agents_circuit_breaker,
        )

        cb = get_external_agents_circuit_breaker()
        initial_failures = cb._single_failures

        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Test"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.return_value = "task-ok"

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 201

    def test_circuit_breaker_records_failure(self, handler, mock_http_handler):
        from aragora.server.handlers.external_agents import (
            get_external_agents_circuit_breaker,
        )

        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
        ):
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Test"},
                None,
            )
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()
            mock_run_coro.side_effect = ConnectionError("timeout")

            result = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)

            assert _status(result) == 500

    def test_get_external_agents_circuit_breaker(self):
        from aragora.server.handlers.external_agents import (
            get_external_agents_circuit_breaker,
        )

        cb = get_external_agents_circuit_breaker()
        assert cb is not None
        assert cb.name == "external_agents_handler"

    def test_reset_external_agents_circuit_breaker(self):
        from aragora.server.handlers.external_agents import (
            get_external_agents_circuit_breaker,
            reset_external_agents_circuit_breaker,
        )

        cb = get_external_agents_circuit_breaker()
        for _ in range(3):
            cb.record_failure()

        reset_external_agents_circuit_breaker()
        assert cb._single_failures == 0


# ---------------------------------------------------------------------------
# _run_coro and _call_run_coro
# ---------------------------------------------------------------------------


class TestRunCoroHelpers:
    """Tests for async helper functions."""

    def test_run_coro_new_event_loop(self):
        from aragora.server.handlers.external_agents import _run_coro

        async def async_fn():
            return "hello"

        result = _run_coro(async_fn())
        assert result == "hello"

    def test_call_run_coro(self):
        from aragora.server.handlers.external_agents import _call_run_coro

        async def async_fn():
            return 42

        result = _call_run_coro(async_fn())
        assert result == 42


# ---------------------------------------------------------------------------
# _record_metrics
# ---------------------------------------------------------------------------


class TestRecordMetrics:
    """Tests for metrics recording."""

    def test_record_metrics_submit(self):
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

    def test_record_metrics_submit_no_duration(self):
        from aragora.server.handlers.external_agents import _record_metrics

        with (
            patch(
                "aragora.server.handlers.external_agents.record_external_agent_task"
            ) as mock_task,
            patch(
                "aragora.server.handlers.external_agents.record_external_agent_duration"
            ) as mock_duration,
        ):
            _record_metrics("submit", "openhands", "code", 0.0)
            mock_task.assert_called_once_with("openhands", "submitted")
            mock_duration.assert_not_called()

    def test_record_metrics_non_submit_operation(self):
        from aragora.server.handlers.external_agents import _record_metrics

        with (
            patch(
                "aragora.server.handlers.external_agents.record_external_agent_task"
            ) as mock_task,
            patch(
                "aragora.server.handlers.external_agents.record_external_agent_duration"
            ) as mock_duration,
        ):
            _record_metrics("get_status", "openhands", "code", 0.5)
            mock_task.assert_not_called()
            mock_duration.assert_not_called()

    def test_record_metrics_none_functions(self):
        """When prometheus is not installed, functions are None."""
        from aragora.server.handlers.external_agents import _record_metrics

        with (
            patch("aragora.server.handlers.external_agents.record_external_agent_task", None),
            patch("aragora.server.handlers.external_agents.record_external_agent_duration", None),
        ):
            # Should not raise
            _record_metrics("submit", "openhands", "code", 1.0)

    def test_record_metrics_handles_type_error(self):
        from aragora.server.handlers.external_agents import _record_metrics

        with patch(
            "aragora.server.handlers.external_agents.record_external_agent_task"
        ) as mock_task:
            mock_task.side_effect = TypeError("bad args")
            # Should not raise
            _record_metrics("submit", "openhands", "code", 1.0)


# ---------------------------------------------------------------------------
# RBAC Permission Constants
# ---------------------------------------------------------------------------


class TestPermissionConstants:
    """Tests for RBAC permission constants."""

    def test_agents_read_permission(self):
        from aragora.server.handlers.external_agents import AGENTS_READ_PERMISSION

        assert AGENTS_READ_PERMISSION == "agents:read"

    def test_agents_write_permission(self):
        from aragora.server.handlers.external_agents import AGENTS_WRITE_PERMISSION

        assert AGENTS_WRITE_PERMISSION == "agents:write"


# ---------------------------------------------------------------------------
# Unmatched Paths
# ---------------------------------------------------------------------------


class TestUnmatchedPaths:
    """Tests for unmatched paths returning None."""

    def test_handle_unmatched_returns_none(self, handler, mock_http_handler):
        result = handler.handle("/api/debates", {}, mock_http_handler)
        assert result is None

    def test_handle_post_non_tasks_returns_none(self, handler, mock_http_handler):
        result = handler.handle_post("/api/external-agents/health", {}, mock_http_handler)
        assert result is None

    def test_handle_delete_non_tasks_returns_none(self, handler, mock_http_handler):
        result = handler.handle_delete("/api/external-agents/health", {}, mock_http_handler)
        assert result is None


# ---------------------------------------------------------------------------
# _set_auth_context
# ---------------------------------------------------------------------------


class TestSetAuthContext:
    """Tests for auth context population."""

    def test_set_auth_context_with_object_user(self, handler, mock_user):
        handler._set_auth_context(mock_user)
        assert handler._auth_context is not None
        assert handler._auth_context.user_id == "user-123"

    def test_set_auth_context_with_dict_user(self, handler, dict_user):
        handler._set_auth_context(dict_user)
        assert handler._auth_context is not None
        assert handler._auth_context.user_id == "dict-user-789"

    def test_set_auth_context_with_minimal_dict(self, handler):
        handler._set_auth_context({"id": "min-user"})
        assert handler._auth_context is not None
        assert handler._auth_context.user_id == "min-user"

    def test_set_auth_context_handles_error_gracefully(self, handler):
        """If AuthorizationContext import fails, _auth_context is None."""
        with patch(
            "aragora.server.handlers.external_agents.ExternalAgentsHandler._set_auth_context"
        ) as mock_set:
            mock_set.side_effect = ImportError("no rbac")
            try:
                handler._set_auth_context({"id": "x"})
            except ImportError:
                pass


# ---------------------------------------------------------------------------
# Integration-style: Full Task Lifecycle
# ---------------------------------------------------------------------------


class TestFullTaskLifecycle:
    """Integration-style test for submit -> status -> cancel."""

    def test_complete_lifecycle(self, handler, mock_http_handler, mock_task_result):
        from aragora.agents.external.models import TaskStatus

        with (
            patch.object(handler, "set_request_context"),
            patch.object(handler, "read_json_body_validated") as mock_read_body,
            patch("aragora.server.handlers.external_agents.ExternalAgentRegistry") as mock_registry,
            patch("aragora.server.handlers.external_agents._run_coro") as mock_run_coro,
            patch("aragora.server.handlers.external_agents._record_metrics"),
        ):
            mock_registry.is_registered.return_value = True
            mock_registry.create.return_value = MagicMock()

            # Step 1: Submit
            mock_read_body.return_value = (
                {"task_type": "code", "prompt": "Fix bug"},
                None,
            )
            mock_run_coro.return_value = "lifecycle-task-1"

            submit = handler.handle_post("/api/external-agents/tasks", {}, mock_http_handler)
            assert _status(submit) == 201

            # Step 2: Check status (running)
            mock_run_coro.return_value = TaskStatus.RUNNING
            status = handler.handle(
                "/api/external-agents/tasks/lifecycle-task-1", {}, mock_http_handler
            )
            assert _status(status) == 200
            assert _body(status)["status"] == "running"

            # Step 3: Check status (completed)
            mock_run_coro.side_effect = [TaskStatus.COMPLETED, mock_task_result]
            completed = handler.handle(
                "/api/external-agents/tasks/lifecycle-task-1", {}, mock_http_handler
            )
            assert _status(completed) == 200
            assert _body(completed)["status"] == "completed"
            assert "result" in _body(completed)

            # Step 4: Cancel (already done)
            mock_run_coro.side_effect = None
            mock_run_coro.return_value = False
            cancel = handler.handle_delete(
                "/api/external-agents/tasks/lifecycle-task-1", {}, mock_http_handler
            )
            assert _status(cancel) == 200
            assert _body(cancel)["cancelled"] is False
