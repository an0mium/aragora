"""Comprehensive tests for memory coordinator handler.

Covers all routes, RBAC, rate limiting, error handling, and edge cases
for aragora/server/handlers/memory/coordinator.py (197 lines).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.memory.coordinator import (
    COORDINATOR_AVAILABLE,
    COORDINATOR_PERMISSION,
    MEMORY_READ_PERMISSION,
    MEMORY_WRITE_PERMISSION,
    CoordinatorHandler,
    _coordinator_limiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for coordinator tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        client_ip: str = "127.0.0.1",
    ):
        self.command = method
        self.client_address = (client_ip, 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


def _make_mock_coordinator(
    metrics: dict | None = None,
    continuum: bool = True,
    consensus: bool = True,
    critique: bool = False,
    mound: bool = False,
    rollback_keys: list[str] | None = None,
    options: Any = None,
) -> MagicMock:
    """Build a mock MemoryCoordinator with configurable subsystems."""
    mock = MagicMock()
    mock.get_metrics.return_value = (
        metrics
        if metrics is not None
        else {
            "total_transactions": 50,
            "successful_transactions": 48,
            "partial_failures": 2,
            "rollbacks_performed": 1,
            "success_rate": 0.96,
        }
    )
    mock.continuum_memory = MagicMock() if continuum else None
    mock.consensus_memory = MagicMock() if consensus else None
    mock.critique_store = MagicMock() if critique else None
    mock.knowledge_mound = MagicMock() if mound else None
    mock._rollback_handlers = {k: lambda x: x for k in (rollback_keys or [])}
    if options is not None:
        mock.options = options
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create handler with empty context."""
    return CoordinatorHandler(server_context={})


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiter state between tests."""
    _coordinator_limiter._buckets.clear()
    yield
    _coordinator_limiter._buckets.clear()


# ===========================================================================
# 1. can_handle routing
# ===========================================================================


class TestCanHandle:
    """Test route matching via can_handle()."""

    def test_metrics_route(self, handler):
        assert handler.can_handle("/api/v1/memory/coordinator/metrics") is True

    def test_config_route(self, handler):
        assert handler.can_handle("/api/v1/memory/coordinator/config") is True

    def test_unknown_route_rejected(self, handler):
        assert handler.can_handle("/api/v1/memory/coordinator/unknown") is False

    def test_unversioned_route_rejected(self, handler):
        assert handler.can_handle("/api/memory/coordinator/metrics") is False

    def test_partial_path_rejected(self, handler):
        assert handler.can_handle("/api/v1/memory/coordinator") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_trailing_slash_rejected(self, handler):
        assert handler.can_handle("/api/v1/memory/coordinator/metrics/") is False

    def test_routes_class_attribute(self, handler):
        assert len(handler.ROUTES) == 2
        assert "/api/v1/memory/coordinator/metrics" in handler.ROUTES
        assert "/api/v1/memory/coordinator/config" in handler.ROUTES


# ===========================================================================
# 2. Initialization
# ===========================================================================


class TestInit:
    """Test handler construction and context handling."""

    def test_init_with_server_context(self):
        ctx = {"memory_coordinator": MagicMock()}
        h = CoordinatorHandler(server_context=ctx)
        assert h.ctx is ctx

    def test_init_with_ctx_param(self):
        ctx = {"some": "value"}
        h = CoordinatorHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_init_server_context_takes_precedence(self):
        ctx1 = {"ctx_param": True}
        ctx2 = {"server_context_param": True}
        h = CoordinatorHandler(ctx=ctx1, server_context=ctx2)
        assert h.ctx is ctx2

    def test_init_no_args_defaults_to_empty(self):
        h = CoordinatorHandler()
        assert h.ctx == {}


# ===========================================================================
# 3. _get_coordinator
# ===========================================================================


class TestGetCoordinator:
    """Test coordinator extraction from context."""

    def test_returns_coordinator_when_present(self):
        mock = MagicMock()
        h = CoordinatorHandler(server_context={"memory_coordinator": mock})
        assert h._get_coordinator() is mock

    def test_returns_none_when_missing(self, handler):
        assert handler._get_coordinator() is None

    def test_returns_none_when_none_in_context(self):
        h = CoordinatorHandler(server_context={"memory_coordinator": None})
        assert h._get_coordinator() is None


# ===========================================================================
# 4. Metrics endpoint (GET /api/v1/memory/coordinator/metrics)
# ===========================================================================


class TestMetricsEndpoint:
    """Test _get_metrics and the metrics route."""

    @pytest.mark.asyncio
    async def test_default_metrics_no_coordinator(self, handler):
        """Returns default zeros when no coordinator configured."""
        result = await handler.handle("/api/v1/memory/coordinator/metrics", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["configured"] is False
        assert body["metrics"]["total_transactions"] == 0
        assert body["metrics"]["successful_transactions"] == 0
        assert body["metrics"]["partial_failures"] == 0
        assert body["metrics"]["rollbacks_performed"] == 0
        assert body["metrics"]["success_rate"] == 0.0
        for sys_name in ("continuum", "consensus", "critique", "mound"):
            assert body["memory_systems"][sys_name] is False

    @pytest.mark.asyncio
    async def test_metrics_with_coordinator(self):
        """Returns real metrics from coordinator."""
        mock_coord = _make_mock_coordinator(
            metrics={
                "total_transactions": 200,
                "successful_transactions": 190,
                "partial_failures": 8,
                "rollbacks_performed": 5,
                "success_rate": 0.95,
            },
            continuum=True,
            consensus=True,
            critique=True,
            mound=True,
            rollback_keys=["continuum", "consensus", "critique"],
        )
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/metrics", {})

        assert _status(result) == 200
        body = _body(result)
        assert body["configured"] is True
        assert body["metrics"]["total_transactions"] == 200
        assert body["metrics"]["successful_transactions"] == 190
        assert body["metrics"]["partial_failures"] == 8
        assert body["metrics"]["rollbacks_performed"] == 5
        assert body["metrics"]["success_rate"] == 0.95
        assert body["memory_systems"]["continuum"] is True
        assert body["memory_systems"]["consensus"] is True
        assert body["memory_systems"]["critique"] is True
        assert body["memory_systems"]["mound"] is True
        assert set(body["rollback_handlers"]) == {"continuum", "consensus", "critique"}

    @pytest.mark.asyncio
    async def test_metrics_partial_subsystems(self):
        """Some memory systems enabled, some not."""
        mock_coord = _make_mock_coordinator(
            continuum=True, consensus=False, critique=False, mound=True
        )
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/metrics", {})

        body = _body(result)
        assert body["memory_systems"]["continuum"] is True
        assert body["memory_systems"]["consensus"] is False
        assert body["memory_systems"]["critique"] is False
        assert body["memory_systems"]["mound"] is True

    @pytest.mark.asyncio
    async def test_metrics_empty_rollback_handlers(self):
        """No rollback handlers registered."""
        mock_coord = _make_mock_coordinator(rollback_keys=[])
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/metrics", {})

        body = _body(result)
        assert body["rollback_handlers"] == []

    @pytest.mark.asyncio
    async def test_metrics_missing_keys_use_defaults(self):
        """Metrics dict missing keys falls back to 0."""
        mock_coord = _make_mock_coordinator(metrics={})
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/metrics", {})

        body = _body(result)
        assert body["metrics"]["total_transactions"] == 0
        assert body["metrics"]["successful_transactions"] == 0
        assert body["metrics"]["partial_failures"] == 0
        assert body["metrics"]["rollbacks_performed"] == 0
        assert body["metrics"]["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_coordinator_error_handled(self):
        """Exception in get_metrics() is caught by @handle_errors."""
        mock_coord = MagicMock()
        mock_coord.get_metrics.side_effect = RuntimeError("DB connection failed")
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/metrics", {})

        assert _status(result) == 500
        body = _body(result)
        assert "error" in body


# ===========================================================================
# 5. Config endpoint (GET /api/v1/memory/coordinator/config)
# ===========================================================================


class TestConfigEndpoint:
    """Test _get_config and the config route."""

    @pytest.mark.asyncio
    async def test_default_config_no_coordinator(self, handler):
        """Returns defaults when no coordinator configured."""
        result = await handler.handle("/api/v1/memory/coordinator/config", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["configured"] is False
        # Should have an options dict (either from CoordinatorOptions defaults or empty)
        assert "options" in body

    @pytest.mark.asyncio
    async def test_config_with_coordinator_and_options(self):
        """Returns full options from coordinator."""
        from aragora.memory.coordinator import CoordinatorOptions

        opts = CoordinatorOptions()
        opts.write_continuum = True
        opts.write_consensus = False
        opts.write_critique = True
        opts.write_mound = False
        opts.rollback_on_failure = False
        opts.parallel_writes = True
        opts.min_confidence_for_mound = 0.9

        mock_coord = MagicMock()
        mock_coord.options = opts
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/config", {})

        assert _status(result) == 200
        body = _body(result)
        assert body["configured"] is True
        assert body["options"]["write_continuum"] is True
        assert body["options"]["write_consensus"] is False
        assert body["options"]["write_critique"] is True
        assert body["options"]["write_mound"] is False
        assert body["options"]["rollback_on_failure"] is False
        assert body["options"]["parallel_writes"] is True
        assert body["options"]["min_confidence_for_mound"] == 0.9

    @pytest.mark.asyncio
    async def test_config_coordinator_without_options_attr(self):
        """Coordinator without .options attribute falls back to defaults."""
        from aragora.memory.coordinator import CoordinatorOptions

        mock_coord = MagicMock(spec=[])  # no attributes at all
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/config", {})

        assert _status(result) == 200
        body = _body(result)
        assert body["configured"] is True
        # Falls back to CoordinatorOptions() defaults
        defaults = CoordinatorOptions()
        assert body["options"]["write_continuum"] == defaults.write_continuum
        assert body["options"]["rollback_on_failure"] == defaults.rollback_on_failure

    @pytest.mark.asyncio
    async def test_config_all_options_present(self):
        """All 7 option keys are always returned."""
        from aragora.memory.coordinator import CoordinatorOptions

        mock_coord = MagicMock()
        mock_coord.options = CoordinatorOptions()
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/config", {})

        body = _body(result)
        expected_keys = {
            "write_continuum",
            "write_consensus",
            "write_critique",
            "write_mound",
            "rollback_on_failure",
            "parallel_writes",
            "min_confidence_for_mound",
        }
        assert set(body["options"].keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_config_error_handled(self):
        """Exception during config retrieval returns 500."""
        mock_coord = MagicMock()
        # options property raises
        type(mock_coord).options = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("corrupt config"))
        )
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/config", {})

        assert _status(result) == 500
        body = _body(result)
        assert "error" in body


# ===========================================================================
# 6. Unknown path returns None
# ===========================================================================


class TestUnknownPath:
    """Test that unrecognized paths return None from handle()."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, handler):
        """Paths not in ROUTES return None after passing auth."""
        result = await handler.handle("/api/v1/memory/coordinator/other", {})
        assert result is None


# ===========================================================================
# 7. Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting behavior on coordinator endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_metrics(self, handler):
        with patch.object(_coordinator_limiter, "is_allowed", return_value=False):
            result = await handler.handle(
                "/api/v1/memory/coordinator/metrics", {}, MockHTTPHandler()
            )
        assert _status(result) == 429
        body = _body(result)
        assert "Rate limit" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_config(self, handler):
        with patch.object(_coordinator_limiter, "is_allowed", return_value=False):
            result = await handler.handle(
                "/api/v1/memory/coordinator/config", {}, MockHTTPHandler()
            )
        assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_rate_limit_allowed_passes_through(self, handler):
        with patch.object(_coordinator_limiter, "is_allowed", return_value=True):
            result = await handler.handle(
                "/api/v1/memory/coordinator/metrics", {}, MockHTTPHandler()
            )
        assert _status(result) == 200


# ===========================================================================
# 8. RBAC / Authentication (opt-out of auto-auth)
# ===========================================================================


class TestRBAC:
    """Test authentication and permission enforcement."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self):
        """Missing auth returns 401."""
        from aragora.server.handlers.utils.auth import UnauthorizedError

        h = CoordinatorHandler(server_context={})

        async def raise_unauth(self, handler, require_auth=True):
            raise UnauthorizedError("No token")

        with patch.object(type(h), "get_auth_context", raise_unauth):
            result = await h.handle("/api/v1/memory/coordinator/metrics", {}, MockHTTPHandler())
        assert _status(result) == 401
        body = _body(result)
        assert "Authentication required" in body.get("error", "")

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self):
        """Insufficient permissions returns 403."""
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import ForbiddenError

        h = CoordinatorHandler(server_context={})

        mock_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="u@example.com",
            org_id="org-1",
            roles={"viewer"},
            permissions={"memory:read"},
        )

        async def return_ctx(self, handler, require_auth=True):
            return mock_ctx

        def deny_permission(self, auth_ctx, perm, resource_id=None):
            raise ForbiddenError("Permission denied", permission=perm)

        with (
            patch.object(type(h), "get_auth_context", return_ctx),
            patch.object(type(h), "check_permission", deny_permission),
        ):
            result = await h.handle("/api/v1/memory/coordinator/metrics", {}, MockHTTPHandler())
        assert _status(result) == 403
        body = _body(result)
        assert "Permission denied" in body.get("error", "")


# ===========================================================================
# 9. COORDINATOR_AVAILABLE flag
# ===========================================================================


class TestCoordinatorAvailable:
    """Test behavior when the coordinator module is unavailable."""

    @pytest.mark.asyncio
    async def test_coordinator_unavailable_returns_501(self, handler):
        with patch("aragora.server.handlers.memory.coordinator.COORDINATOR_AVAILABLE", False):
            result = await handler.handle(
                "/api/v1/memory/coordinator/metrics", {}, MockHTTPHandler()
            )
        assert _status(result) == 501
        body = _body(result)
        assert "not available" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_coordinator_unavailable_config(self, handler):
        with patch("aragora.server.handlers.memory.coordinator.COORDINATOR_AVAILABLE", False):
            result = await handler.handle(
                "/api/v1/memory/coordinator/config", {}, MockHTTPHandler()
            )
        assert _status(result) == 501


# ===========================================================================
# 10. Permission constants
# ===========================================================================


class TestPermissionConstants:
    """Verify permission string constants are correct."""

    def test_coordinator_permission(self):
        assert COORDINATOR_PERMISSION == "memory:admin"

    def test_write_permission(self):
        assert MEMORY_WRITE_PERMISSION == "memory:write"

    def test_read_permission(self):
        assert MEMORY_READ_PERMISSION == "memory:read"


# ===========================================================================
# 11. Edge cases & query_params
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge and corner cases."""

    @pytest.mark.asyncio
    async def test_query_params_ignored_by_metrics(self, handler):
        """Query params have no effect on metrics endpoint."""
        result = await handler.handle(
            "/api/v1/memory/coordinator/metrics",
            {"format": "csv", "verbose": "true"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_query_params_ignored_by_config(self, handler):
        result = await handler.handle(
            "/api/v1/memory/coordinator/config",
            {"debug": "1"},
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handler_arg_none(self, handler):
        """handle() with handler=None (no HTTP handler)."""
        result = await handler.handle("/api/v1/memory/coordinator/metrics", {}, None)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_metrics_coordinator_zero_transactions(self):
        """Coordinator with zero transactions."""
        mock_coord = _make_mock_coordinator(
            metrics={
                "total_transactions": 0,
                "successful_transactions": 0,
                "partial_failures": 0,
                "rollbacks_performed": 0,
                "success_rate": 0.0,
            },
        )
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/metrics", {})

        body = _body(result)
        assert body["configured"] is True
        assert body["metrics"]["total_transactions"] == 0
        assert body["metrics"]["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_extra_keys_in_metrics_dict(self):
        """Extra keys in metrics dict are not exposed."""
        mock_coord = _make_mock_coordinator(
            metrics={
                "total_transactions": 10,
                "successful_transactions": 9,
                "partial_failures": 1,
                "rollbacks_performed": 0,
                "success_rate": 0.9,
                "extra_internal_key": "should_not_appear",
            },
        )
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/metrics", {})

        body = _body(result)
        assert "extra_internal_key" not in body["metrics"]

    @pytest.mark.asyncio
    async def test_multiple_rollback_handlers(self):
        """Multiple rollback handler keys appear in response."""
        mock_coord = _make_mock_coordinator(
            rollback_keys=["continuum", "consensus", "critique", "mound"]
        )
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/metrics", {})

        body = _body(result)
        assert len(body["rollback_handlers"]) == 4

    @pytest.mark.asyncio
    async def test_config_default_values_match_coordinator_options(self):
        """Default config matches CoordinatorOptions default attributes."""
        from aragora.memory.coordinator import CoordinatorOptions

        mock_coord = MagicMock()
        mock_coord.options = CoordinatorOptions()
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/config", {})

        body = _body(result)
        opts = body["options"]
        assert opts["write_continuum"] is True
        assert opts["write_consensus"] is True
        assert opts["write_critique"] is True
        assert opts["write_mound"] is True
        assert opts["rollback_on_failure"] is True
        assert opts["parallel_writes"] is False
        assert opts["min_confidence_for_mound"] == 0.7

    @pytest.mark.asyncio
    async def test_config_min_confidence_boundary(self):
        """Min confidence at 0.0 and 1.0."""
        from aragora.memory.coordinator import CoordinatorOptions

        opts = CoordinatorOptions()
        opts.min_confidence_for_mound = 0.0
        mock_coord = MagicMock()
        mock_coord.options = opts
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})
        result = await h.handle("/api/v1/memory/coordinator/config", {})
        body = _body(result)
        assert body["options"]["min_confidence_for_mound"] == 0.0

        opts.min_confidence_for_mound = 1.0
        result = await h.handle("/api/v1/memory/coordinator/config", {})
        body = _body(result)
        assert body["options"]["min_confidence_for_mound"] == 1.0

    @pytest.mark.asyncio
    async def test_config_no_coordinator_with_coordinator_options_available(self, handler):
        """No coordinator but CoordinatorOptions import succeeds yields its defaults."""
        result = await handler.handle("/api/v1/memory/coordinator/config", {})
        body = _body(result)
        assert body["configured"] is False
        # options is CoordinatorOptions().__dict__ (non-empty dict)
        assert isinstance(body["options"], dict)

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_paths(self):
        """Both routes work when called on same handler instance."""
        mock_coord = _make_mock_coordinator(rollback_keys=["x"])
        h = CoordinatorHandler(server_context={"memory_coordinator": mock_coord})

        r1 = await h.handle("/api/v1/memory/coordinator/metrics", {})
        r2 = await h.handle("/api/v1/memory/coordinator/config", {})

        assert _status(r1) == 200
        assert _status(r2) == 200
        assert _body(r1)["configured"] is True
        assert _body(r2)["configured"] is True
