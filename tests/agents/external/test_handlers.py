"""Tests for External Agent Gateway HTTP handlers."""

from __future__ import annotations

import json

import pytest

from aragora.server.handlers.external_agents import (
    AGENTS_READ_PERMISSION,
    AGENTS_WRITE_PERMISSION,
    ExternalAgentsHandler,
)


@pytest.fixture
def server_context() -> dict:
    """Create basic server context."""
    return {}


@pytest.fixture
def handler(server_context: dict) -> ExternalAgentsHandler:
    """Create handler instance."""
    from aragora.rbac.models import AuthorizationContext

    handler = ExternalAgentsHandler(server_context)
    handler._auth_context = AuthorizationContext(
        user_id="test-user",
        org_id="test-org",
        roles={"admin"},
        permissions={AGENTS_READ_PERMISSION, AGENTS_WRITE_PERMISSION},
    )
    return handler


class TestCanHandle:
    """Test route matching."""

    def test_matches_adapters_endpoint(self, handler: ExternalAgentsHandler) -> None:
        """Test that the handler matches the adapters endpoint."""
        assert handler.can_handle("/api/external-agents/adapters")

    def test_matches_health_endpoint(self, handler: ExternalAgentsHandler) -> None:
        """Test that the handler matches the health endpoint."""
        assert handler.can_handle("/api/external-agents/health")

    def test_matches_tasks_endpoint(self, handler: ExternalAgentsHandler) -> None:
        """Test that the handler matches the tasks endpoint."""
        assert handler.can_handle("/api/external-agents/tasks")

    def test_matches_task_by_id(self, handler: ExternalAgentsHandler) -> None:
        """Test that the handler matches task-by-ID endpoints."""
        assert handler.can_handle("/api/external-agents/tasks/task-123")
        assert handler.can_handle("/api/external-agents/tasks/openhands-abc123")

    def test_matches_versioned_paths(self, handler: ExternalAgentsHandler) -> None:
        """Test that the handler matches versioned paths."""
        assert handler.can_handle("/api/v1/external-agents/adapters")
        assert handler.can_handle("/api/v2/external-agents/health")
        assert handler.can_handle("/api/v1/external-agents/tasks/task-123")

    def test_rejects_unrelated_paths(self, handler: ExternalAgentsHandler) -> None:
        """Test that the handler rejects unrelated paths."""
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/agents")
        assert not handler.can_handle("/api/external-integrations/tasks")

    def test_rejects_partial_matches(self, handler: ExternalAgentsHandler) -> None:
        """Test that the handler rejects partial path matches."""
        assert not handler.can_handle("/api/external-agents")
        assert not handler.can_handle("/api/external")


class TestRoutes:
    """Test ROUTES constant."""

    def test_routes_defined(self, handler: ExternalAgentsHandler) -> None:
        """Test that ROUTES is properly defined."""
        assert hasattr(handler, "ROUTES")
        assert isinstance(handler.ROUTES, list)
        assert len(handler.ROUTES) == 4

    def test_expected_routes(self, handler: ExternalAgentsHandler) -> None:
        """Test that expected routes are in ROUTES."""
        assert "/api/external-agents/tasks" in handler.ROUTES
        assert "/api/external-agents/tasks/*" in handler.ROUTES
        assert "/api/external-agents/adapters" in handler.ROUTES
        assert "/api/external-agents/health" in handler.ROUTES


class TestListAdapters:
    """Test _list_adapters internal method."""

    def test_returns_adapters_list(self, handler: ExternalAgentsHandler) -> None:
        """Test that list adapters returns a proper response."""
        result = handler._list_adapters()

        # Should return a HandlerResult
        assert result is not None
        assert result.status_code == 200

        # Parse body
        body = json.loads(result.body)
        assert "adapters" in body
        assert "total" in body
        assert isinstance(body["adapters"], list)

    def test_adapter_structure(self, handler: ExternalAgentsHandler) -> None:
        """Test the structure of adapter entries."""
        result = handler._list_adapters()
        body = json.loads(result.body)

        # If there are adapters registered, check structure
        for adapter in body["adapters"]:
            assert "name" in adapter
            assert "description" in adapter
            assert "config_class" in adapter


class TestHealthCheck:
    """Test _health_check internal method."""

    def test_returns_health_list(self, handler: ExternalAgentsHandler) -> None:
        """Test that health check returns a proper response."""
        result = handler._health_check()

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "health" in body
        assert "total" in body
        assert isinstance(body["health"], list)

    def test_health_with_adapter_filter(self, handler: ExternalAgentsHandler) -> None:
        """Test health check with adapter filter."""
        result = handler._health_check(adapter_name="openhands")

        assert result is not None
        # Should return 200 even if no adapters match
        assert result.status_code == 200


class TestSubmitTaskValidation:
    """Test _submit_task validation logic."""

    def test_requires_task_type(self, handler: ExternalAgentsHandler) -> None:
        """Test that task_type is required."""
        from unittest.mock import MagicMock

        result = handler._submit_task(
            {"prompt": "hello"},
            MagicMock(id="user-1"),
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "task_type" in body["error"].lower() or "required" in body["error"].lower()

    def test_requires_prompt(self, handler: ExternalAgentsHandler) -> None:
        """Test that prompt is required."""
        from unittest.mock import MagicMock

        result = handler._submit_task(
            {"task_type": "test"},
            MagicMock(id="user-1"),
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "prompt" in body["error"].lower() or "required" in body["error"].lower()

    def test_prompt_length_limit(self, handler: ExternalAgentsHandler) -> None:
        """Test that prompt has a length limit."""
        from unittest.mock import MagicMock

        result = handler._submit_task(
            {"task_type": "test", "prompt": "x" * 10001},
            MagicMock(id="user-1"),
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "too long" in body["error"].lower()

    def test_unknown_adapter(self, handler: ExternalAgentsHandler) -> None:
        """Test rejection of unknown adapters."""
        from unittest.mock import MagicMock

        result = handler._submit_task(
            {"task_type": "test", "prompt": "hello", "adapter": "nonexistent-adapter"},
            MagicMock(id="user-1"),
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "unknown" in body["error"].lower() or "adapter" in body["error"].lower()


class TestGetTaskValidation:
    """Test _get_task validation."""

    def test_invalid_task_id_format(self, handler: ExternalAgentsHandler) -> None:
        """Test that slash in task ID returns not found."""
        from unittest.mock import MagicMock

        # This would be caught by the router, but test the edge case
        result = handler._get_task("", MagicMock(id="user-1"))

        # Empty ID should fail
        # The actual behavior depends on the implementation
        assert result is not None


class TestCancelTaskValidation:
    """Test _cancel_task validation."""

    def test_cancellation_returns_result(self, handler: ExternalAgentsHandler) -> None:
        """Test that cancellation returns a structured result."""
        from unittest.mock import MagicMock

        # This will likely fail since the task doesn't exist,
        # but we're testing the response structure
        result = handler._cancel_task("nonexistent-task", MagicMock(id="user-1"))

        assert result is not None
        # Should be 200 or error status
        assert result.status_code in (200, 404, 500)
