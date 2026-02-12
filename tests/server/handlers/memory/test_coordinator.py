"""Tests for CoordinatorHandler."""

from __future__ import annotations

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


import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.memory.coordinator import (
    CoordinatorHandler,
    COORDINATOR_PERMISSION,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockCoordinatorOptions:
    """Mock coordinator options."""

    write_continuum: bool = True
    write_consensus: bool = True
    write_critique: bool = True
    write_mound: bool = True
    rollback_on_failure: bool = True
    parallel_writes: bool = True
    min_confidence_for_mound: float = 0.5


@dataclass
class MockCoordinator:
    """Mock MemoryCoordinator."""

    continuum_memory: Any | None = None
    consensus_memory: Any | None = None
    critique_store: Any | None = None
    knowledge_mound: Any | None = None
    options: MockCoordinatorOptions = field(default_factory=MockCoordinatorOptions)
    _rollback_handlers: dict[str, Any] = field(default_factory=dict)

    def get_metrics(self) -> dict[str, Any]:
        return {
            "total_transactions": 100,
            "successful_transactions": 95,
            "partial_failures": 3,
            "rollbacks_performed": 2,
            "success_rate": 0.95,
        }


class MockAuthContext:
    """Mock authentication context."""

    def __init__(self, user_id: str = "user-123", permissions: list = None):
        self.user_id = user_id
        self.permissions = permissions or [COORDINATOR_PERMISSION]


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, client_ip: str = "127.0.0.1"):
        self.headers = {"X-Forwarded-For": client_ip}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator."""
    return MockCoordinator(
        continuum_memory=MagicMock(),
        consensus_memory=MagicMock(),
        critique_store=MagicMock(),
        knowledge_mound=MagicMock(),
        _rollback_handlers={"continuum": MagicMock(), "consensus": MagicMock()},
    )


@pytest.fixture
def handler(mock_coordinator):
    """Create a test handler with mock coordinator."""
    ctx = {"memory_coordinator": mock_coordinator}
    h = CoordinatorHandler(server_context=ctx)
    return h


@pytest.fixture
def handler_no_coordinator():
    """Create a test handler without coordinator."""
    return CoordinatorHandler(server_context={})


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test Handler Routing
# =============================================================================


class TestHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_metrics_route(self, handler):
        """Test can_handle for metrics route."""
        assert handler.can_handle("/api/v1/memory/coordinator/metrics") is True

    def test_can_handle_config_route(self, handler):
        """Test can_handle for config route."""
        assert handler.can_handle("/api/v1/memory/coordinator/config") is True

    def test_cannot_handle_invalid_route(self, handler):
        """Test can_handle for invalid route."""
        assert handler.can_handle("/api/v1/memory/other") is False

    def test_cannot_handle_partial_match(self, handler):
        """Test can_handle rejects partial matches."""
        assert handler.can_handle("/api/v1/memory/coordinator") is False


# =============================================================================
# Test Get Metrics
# =============================================================================


class TestGetMetrics:
    """Tests for get metrics endpoint."""

    @pytest.mark.asyncio
    async def test_get_metrics_success(self, handler, mock_coordinator):
        """Test successful metrics retrieval."""
        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MockAuthContext()

            result = handler._get_metrics()

            assert result.status_code == 200
            data = parse_response(result)
            assert data["configured"] is True
            assert data["metrics"]["total_transactions"] == 100
            assert data["metrics"]["success_rate"] == 0.95

    def test_get_metrics_with_memory_systems(self, handler):
        """Test metrics include memory system status."""
        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MockAuthContext()

            result = handler._get_metrics()

            assert result.status_code == 200
            data = parse_response(result)
            assert data["memory_systems"]["continuum"] is True
            assert data["memory_systems"]["consensus"] is True
            assert data["memory_systems"]["critique"] is True
            assert data["memory_systems"]["mound"] is True

    def test_get_metrics_with_rollback_handlers(self, handler):
        """Test metrics include rollback handlers."""
        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MockAuthContext()

            result = handler._get_metrics()

            assert result.status_code == 200
            data = parse_response(result)
            assert "rollback_handlers" in data
            assert "continuum" in data["rollback_handlers"]

    def test_get_metrics_no_coordinator(self, handler_no_coordinator):
        """Test metrics when coordinator not configured."""
        with (
            patch.object(handler_no_coordinator, "get_auth_context") as mock_auth,
            patch.object(handler_no_coordinator, "check_permission"),
        ):
            mock_auth.return_value = MockAuthContext()

            result = handler_no_coordinator._get_metrics()

            assert result.status_code == 200
            data = parse_response(result)
            assert data["configured"] is False
            assert data["metrics"]["total_transactions"] == 0


# =============================================================================
# Test Get Config
# =============================================================================


class TestGetConfig:
    """Tests for get config endpoint."""

    def test_get_config_success(self, handler):
        """Test successful config retrieval."""
        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MockAuthContext()

            result = handler._get_config()

            assert result.status_code == 200
            data = parse_response(result)
            assert data["configured"] is True
            assert data["options"]["write_continuum"] is True
            assert data["options"]["rollback_on_failure"] is True

    def test_get_config_all_options(self, handler):
        """Test config includes all options."""
        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MockAuthContext()

            result = handler._get_config()

            assert result.status_code == 200
            data = parse_response(result)
            options = data["options"]
            assert "write_continuum" in options
            assert "write_consensus" in options
            assert "write_critique" in options
            assert "write_mound" in options
            assert "rollback_on_failure" in options
            assert "parallel_writes" in options
            assert "min_confidence_for_mound" in options

    def test_get_config_no_coordinator(self, handler_no_coordinator):
        """Test config when coordinator not configured."""
        with (
            patch.object(handler_no_coordinator, "get_auth_context") as mock_auth,
            patch.object(handler_no_coordinator, "check_permission"),
        ):
            mock_auth.return_value = MockAuthContext()

            result = handler_no_coordinator._get_config()

            assert result.status_code == 200
            data = parse_response(result)
            assert data["configured"] is False


# =============================================================================
# Test Authentication & Authorization
# =============================================================================


class TestAuthentication:
    """Tests for authentication and authorization."""

    @pytest.mark.asyncio
    async def test_requires_authentication(self, handler):
        """Test endpoint requires authentication."""
        from aragora.server.handlers.secure import UnauthorizedError

        with patch.object(handler, "get_auth_context") as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle("/api/v1/memory/coordinator/metrics", {}, MockHandler())

            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_requires_permission(self, handler):
        """Test endpoint requires coordinator permission."""
        from aragora.server.handlers.secure import ForbiddenError

        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission") as mock_perm,
        ):
            mock_auth.return_value = MockAuthContext(permissions=[])
            mock_perm.side_effect = ForbiddenError("Missing permission")

            result = await handler.handle("/api/v1/memory/coordinator/metrics", {}, MockHandler())

            assert result.status_code == 403


# =============================================================================
# Test Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_initial_requests(self, handler):
        """Test rate limiter allows initial requests."""
        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.memory.coordinator._coordinator_limiter"
            ) as mock_limiter,
        ):
            mock_auth.return_value = MockAuthContext()
            mock_limiter.is_allowed.return_value = True

            result = await handler.handle("/api/v1/memory/coordinator/metrics", {}, MockHandler())

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, handler):
        """Test rate limiter rejects excessive requests."""
        with patch(
            "aragora.server.handlers.memory.coordinator._coordinator_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            result = await handler.handle("/api/v1/memory/coordinator/metrics", {}, MockHandler())

            assert result.status_code == 429
            assert "Rate limit" in parse_response(result)["error"]


# =============================================================================
# Test Internal Method
# =============================================================================


class TestGetCoordinator:
    """Tests for _get_coordinator method."""

    def test_get_coordinator_returns_coordinator(self, handler, mock_coordinator):
        """Test _get_coordinator returns coordinator from context."""
        result = handler._get_coordinator()
        assert result is mock_coordinator

    def test_get_coordinator_returns_none_when_missing(self, handler_no_coordinator):
        """Test _get_coordinator returns None when not in context."""
        result = handler_no_coordinator._get_coordinator()
        assert result is None
