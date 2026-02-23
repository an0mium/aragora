"""
Comprehensive tests for aragora/server/handlers/admin/_health_impl.py.

This module is a backward-compatibility shim that re-exports HealthHandler
and cache utilities from the health/ package. Tests verify:

  TestShimReExports         - All expected symbols are re-exported correctly
  TestHealthHandlerInit     - Constructor and attribute initialization
  TestCanHandle             - Route matching via can_handle()
  TestHandleRouting         - handle() dispatches to correct internal methods
  TestPathNormalization     - v1 and non-v1 paths route to same handler
  TestPublicRoutesBehavior  - Public routes allow unauthenticated access
  TestMinimalHealthCheck    - Unauthenticated /api/health returns minimal response
  TestCacheUtilities        - _get_cached_health / _set_cached_health logic
  TestUnknownPath           - Unknown paths return None

Coverage: re-export validation, routing, auth, caching, edge cases.
Target: 20+ tests, 0 failures.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import through the shim to verify backward-compat path works
from aragora.server.handlers.admin._health_impl import (
    HealthHandler,
    _HEALTH_CACHE,
    _HEALTH_CACHE_TIMESTAMPS,
    _HEALTH_CACHE_TTL,
    _HEALTH_CACHE_TTL_DETAILED,
    _SERVER_START_TIME,
    _get_cached_health,
    _set_cached_health,
)

# Also import from the package to confirm they are the same objects
from aragora.server.handlers.admin.health import (
    HealthHandler as HealthHandlerDirect,
)
from aragora.server.handlers.admin.health import (
    _get_cached_health as _get_cached_health_direct,
)
from aragora.server.handlers.admin.health import (
    _set_cached_health as _set_cached_health_direct,
)


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


def _make_handler(ctx: dict[str, Any] | None = None) -> HealthHandler:
    """Create a HealthHandler with the given context."""
    return HealthHandler(ctx=ctx or {})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Default handler with empty context."""
    return _make_handler()


@pytest.fixture
def http_handler():
    """Mock HTTP handler object passed to handle()."""
    mock = MagicMock()
    mock.command = "GET"
    mock.headers = {"User-Agent": "test-agent"}
    return mock


@pytest.fixture(autouse=True)
def _clear_health_cache():
    """Clear the health cache before each test to avoid cross-test pollution."""
    _HEALTH_CACHE.clear()
    _HEALTH_CACHE_TIMESTAMPS.clear()
    yield
    _HEALTH_CACHE.clear()
    _HEALTH_CACHE_TIMESTAMPS.clear()


# ---------------------------------------------------------------------------
# TestShimReExports
# ---------------------------------------------------------------------------


class TestShimReExports:
    """Verify backward-compatibility shim re-exports the correct objects."""

    def test_health_handler_is_same_class(self):
        """HealthHandler imported via shim is the same class as from the package."""
        assert HealthHandler is HealthHandlerDirect

    def test_get_cached_health_is_same_function(self):
        """_get_cached_health from shim is the same function as from the package."""
        assert _get_cached_health is _get_cached_health_direct

    def test_set_cached_health_is_same_function(self):
        """_set_cached_health from shim is the same function as from the package."""
        assert _set_cached_health is _set_cached_health_direct

    def test_server_start_time_is_number(self):
        """_SERVER_START_TIME is a float representing epoch time."""
        assert isinstance(_SERVER_START_TIME, float)
        assert _SERVER_START_TIME > 0

    def test_health_cache_ttl_values(self):
        """TTL constants have reasonable values."""
        assert isinstance(_HEALTH_CACHE_TTL, (int, float))
        assert _HEALTH_CACHE_TTL > 0
        assert isinstance(_HEALTH_CACHE_TTL_DETAILED, (int, float))
        assert _HEALTH_CACHE_TTL_DETAILED > 0

    def test_all_exports_present(self):
        """__all__ in the shim module lists the expected symbols."""
        from aragora.server.handlers.admin import _health_impl

        expected = {
            "HealthHandler",
            "_get_cached_health",
            "_set_cached_health",
            "_SERVER_START_TIME",
            "_HEALTH_CACHE",
            "_HEALTH_CACHE_TTL",
            "_HEALTH_CACHE_TTL_DETAILED",
            "_HEALTH_CACHE_TIMESTAMPS",
        }
        assert set(_health_impl.__all__) == expected


# ---------------------------------------------------------------------------
# TestHealthHandlerInit
# ---------------------------------------------------------------------------


class TestHealthHandlerInit:
    """Test HealthHandler initialization."""

    def test_default_ctx_is_empty_dict(self):
        """When no ctx is provided, defaults to empty dict."""
        h = HealthHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        """Custom context is stored."""
        ctx = {"storage": MagicMock(), "key": "value"}
        h = HealthHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_becomes_empty_dict(self):
        """Passing None explicitly still produces empty dict."""
        h = HealthHandler(ctx=None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# TestCanHandle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test can_handle() route matching."""

    def test_healthz(self, handler):
        assert handler.can_handle("/healthz") is True

    def test_readyz(self, handler):
        assert handler.can_handle("/readyz") is True

    def test_readyz_dependencies(self, handler):
        assert handler.can_handle("/readyz/dependencies") is True

    def test_api_health(self, handler):
        assert handler.can_handle("/api/health") is True

    def test_api_v1_health(self, handler):
        assert handler.can_handle("/api/v1/health") is True

    def test_api_v1_health_detailed(self, handler):
        assert handler.can_handle("/api/v1/health/detailed") is True

    def test_api_health_workers(self, handler):
        assert handler.can_handle("/api/v1/health/workers") is True

    def test_api_diagnostics(self, handler):
        assert handler.can_handle("/api/diagnostics") is True

    def test_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/unknown") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_match_not_accepted(self, handler):
        """Partial prefixes should not match."""
        assert handler.can_handle("/healthz/extra") is False

    def test_all_routes_are_handleable(self, handler):
        """Every entry in ROUTES should be accepted by can_handle."""
        for route in HealthHandler.ROUTES:
            assert handler.can_handle(route) is True, f"Route not handled: {route}"


# ---------------------------------------------------------------------------
# TestHandleRouting
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Test that handle() dispatches to the correct internal methods."""

    @pytest.mark.asyncio
    async def test_healthz_calls_liveness(self, handler, http_handler):
        """GET /healthz dispatches to _liveness_probe."""
        with patch.object(handler, "_liveness_probe", return_value=MagicMock(status_code=200, body=b'{"status":"ok"}')) as m:
            result = await handler.handle("/healthz", {}, http_handler)
            m.assert_called_once()
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_readyz_calls_readiness_probe(self, handler, http_handler):
        """GET /readyz dispatches to _readiness_probe_fast."""
        with patch.object(handler, "_readiness_probe_fast", return_value=MagicMock(status_code=200, body=b'{"status":"ok"}')) as m:
            result = await handler.handle("/readyz", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_readyz_dependencies_calls_readiness_dependencies(self, handler, http_handler):
        """GET /readyz/dependencies dispatches to _readiness_dependencies."""
        with patch.object(handler, "_readiness_dependencies", return_value=MagicMock(status_code=200, body=b'{}')) as m:
            result = await handler.handle("/readyz/dependencies", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, handler, http_handler):
        """Unknown paths return None."""
        result = await handler.handle("/api/v1/totally-unknown", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_workers_route(self, handler, http_handler):
        """GET /api/v1/health/workers dispatches to _worker_health_status."""
        with patch.object(handler, "_worker_health_status", return_value=MagicMock(status_code=200, body=b'{}')) as m:
            result = await handler.handle("/api/v1/health/workers", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_queue_route(self, handler, http_handler):
        """GET /api/v1/health/job-queue dispatches to _job_queue_health_status."""
        with patch.object(handler, "_job_queue_health_status", return_value=MagicMock(status_code=200, body=b'{}')) as m:
            result = await handler.handle("/api/v1/health/job-queue", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_workers_all_route(self, handler, http_handler):
        """GET /api/v1/health/workers/all dispatches to _combined_worker_queue_health."""
        with patch.object(handler, "_combined_worker_queue_health", return_value=MagicMock(status_code=200, body=b'{}')) as m:
            result = await handler.handle("/api/v1/health/workers/all", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_diagnostics_route(self, handler, http_handler):
        """GET /api/v1/diagnostics dispatches to _deployment_diagnostics."""
        with patch.object(handler, "_deployment_diagnostics", return_value=MagicMock(status_code=200, body=b'{}')) as m:
            result = await handler.handle("/api/v1/diagnostics", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_diagnostics_deployment_route(self, handler, http_handler):
        """GET /api/v1/diagnostics/deployment dispatches to _deployment_diagnostics."""
        with patch.object(handler, "_deployment_diagnostics", return_value=MagicMock(status_code=200, body=b'{}')) as m:
            result = await handler.handle("/api/v1/diagnostics/deployment", {}, http_handler)
            m.assert_called_once()


# ---------------------------------------------------------------------------
# TestPathNormalization
# ---------------------------------------------------------------------------


class TestPathNormalization:
    """Test that v1 and non-v1 paths normalize and route correctly."""

    @pytest.mark.asyncio
    async def test_v1_health_normalizes_to_api_health(self, handler, http_handler):
        """Both /api/v1/health and /api/health route to _health_check (when authenticated)."""
        mock_result = MagicMock(status_code=200, body=b'{"status":"healthy"}')
        with patch.object(handler, "_health_check", return_value=mock_result) as m:
            result = await handler.handle("/api/v1/health", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_v1_health_also_routes(self, handler, http_handler):
        """Non-v1 /api/health also routes to _health_check (when authenticated)."""
        mock_result = MagicMock(status_code=200, body=b'{"status":"healthy"}')
        with patch.object(handler, "_health_check", return_value=mock_result) as m:
            result = await handler.handle("/api/health", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_v1_detailed_normalizes(self, handler, http_handler):
        """/api/v1/health/detailed normalizes to /api/health/detailed."""
        mock_result = MagicMock(status_code=200, body=b'{}')
        with patch.object(handler, "_detailed_health_check", return_value=mock_result) as m:
            result = await handler.handle("/api/v1/health/detailed", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_v1_detailed_also_routes(self, handler, http_handler):
        """/api/health/detailed also routes to _detailed_health_check."""
        mock_result = MagicMock(status_code=200, body=b'{}')
        with patch.object(handler, "_detailed_health_check", return_value=mock_result) as m:
            result = await handler.handle("/api/health/detailed", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_v1_stores_normalizes(self, handler, http_handler):
        """/api/v1/health/stores normalizes correctly."""
        mock_result = MagicMock(status_code=200, body=b'{}')
        with patch.object(handler, "_database_stores_health", return_value=mock_result) as m:
            result = await handler.handle("/api/v1/health/stores", {}, http_handler)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_v1_platform_health_normalizes(self, handler, http_handler):
        """/api/v1/platform/health normalizes to /api/platform/health."""
        mock_result = MagicMock(status_code=200, body=b'{}')
        with patch.object(handler, "_platform_health", return_value=mock_result) as m:
            result = await handler.handle("/api/v1/platform/health", {}, http_handler)
            m.assert_called_once()


# ---------------------------------------------------------------------------
# TestMinimalHealthCheck
# ---------------------------------------------------------------------------


class TestMinimalHealthCheck:
    """Test the _minimal_health_check() method."""

    def test_minimal_returns_healthy_by_default(self, handler):
        """When degraded_mode import fails, returns healthy with 200."""
        with patch.dict("sys.modules", {"aragora.server.degraded_mode": None}):
            result = handler._minimal_health_check()
        body = _body(result)
        assert body["status"] == "healthy"
        assert _status(result) == 200
        assert "timestamp" in body

    def test_minimal_returns_degraded_when_degraded(self, handler):
        """When is_degraded() returns True, returns 503 with degraded status."""
        mock_module = MagicMock()
        mock_module.is_degraded.return_value = True
        with patch.dict("sys.modules", {"aragora.server.degraded_mode": mock_module}):
            result = handler._minimal_health_check()
        body = _body(result)
        assert body["status"] == "degraded"
        assert _status(result) == 503

    def test_minimal_returns_healthy_when_not_degraded(self, handler):
        """When is_degraded() returns False, returns 200 with healthy status."""
        mock_module = MagicMock()
        mock_module.is_degraded.return_value = False
        with patch.dict("sys.modules", {"aragora.server.degraded_mode": mock_module}):
            result = handler._minimal_health_check()
        body = _body(result)
        assert body["status"] == "healthy"
        assert _status(result) == 200

    def test_minimal_has_no_version_info(self, handler):
        """Minimal response should NOT leak version or internal details."""
        result = handler._minimal_health_check()
        body = _body(result)
        assert "version" not in body
        assert "checks" not in body
        assert "uptime" not in body


# ---------------------------------------------------------------------------
# TestCacheUtilities
# ---------------------------------------------------------------------------


class TestCacheUtilities:
    """Test _get_cached_health and _set_cached_health."""

    def test_get_returns_none_when_empty(self):
        """Getting from an empty cache returns None."""
        assert _get_cached_health("test_key") is None

    def test_set_then_get_returns_value(self):
        """Setting a value then getting it returns the stored value."""
        data = {"status": "healthy", "checks": {}}
        _set_cached_health("test_key", data)
        result = _get_cached_health("test_key")
        assert result == data

    def test_cache_expires_after_ttl(self):
        """Cached values expire after TTL."""
        data = {"status": "healthy"}
        _set_cached_health("expire_key", data)
        # Backdate the timestamp so it appears expired
        _HEALTH_CACHE_TIMESTAMPS["expire_key"] = time.time() - _HEALTH_CACHE_TTL - 1
        assert _get_cached_health("expire_key") is None

    def test_cache_valid_within_ttl(self):
        """Cached values are valid within TTL."""
        data = {"status": "ok"}
        _set_cached_health("valid_key", data)
        # Timestamp was set to now, should still be valid
        assert _get_cached_health("valid_key") == data

    def test_multiple_keys_independent(self):
        """Different keys are cached independently."""
        _set_cached_health("key_a", {"a": 1})
        _set_cached_health("key_b", {"b": 2})
        assert _get_cached_health("key_a") == {"a": 1}
        assert _get_cached_health("key_b") == {"b": 2}

    def test_overwrite_existing_key(self):
        """Setting a key that already exists overwrites it."""
        _set_cached_health("overwrite", {"old": True})
        _set_cached_health("overwrite", {"new": True})
        assert _get_cached_health("overwrite") == {"new": True}

    def test_get_with_missing_timestamp_returns_none(self):
        """If cache has data but no timestamp, returns None (treated as expired)."""
        _HEALTH_CACHE["no_ts_key"] = {"data": True}
        # No entry in _HEALTH_CACHE_TIMESTAMPS
        result = _get_cached_health("no_ts_key")
        # time.time() - 0 > TTL, so it's expired
        assert result is None


# ---------------------------------------------------------------------------
# TestPublicRoutes
# ---------------------------------------------------------------------------


class TestPublicRoutes:
    """Test PUBLIC_ROUTES configuration."""

    def test_healthz_is_public(self):
        assert "/healthz" in HealthHandler.PUBLIC_ROUTES

    def test_readyz_is_public(self):
        assert "/readyz" in HealthHandler.PUBLIC_ROUTES

    def test_readyz_dependencies_is_public(self):
        assert "/readyz/dependencies" in HealthHandler.PUBLIC_ROUTES

    def test_api_health_is_public(self):
        assert "/api/health" in HealthHandler.PUBLIC_ROUTES

    def test_api_v1_health_is_public(self):
        assert "/api/v1/health" in HealthHandler.PUBLIC_ROUTES

    def test_detailed_is_not_public(self):
        """Detailed health requires authentication."""
        assert "/api/v1/health/detailed" not in HealthHandler.PUBLIC_ROUTES

    def test_stores_is_not_public(self):
        """Stores health requires authentication."""
        assert "/api/v1/health/stores" not in HealthHandler.PUBLIC_ROUTES


# ---------------------------------------------------------------------------
# TestHandlerConstants
# ---------------------------------------------------------------------------


class TestHandlerConstants:
    """Test handler class-level constants."""

    def test_health_permission(self):
        assert HealthHandler.HEALTH_PERMISSION == "system.health.read"

    def test_resource_type(self):
        assert HealthHandler.RESOURCE_TYPE == "health"

    def test_routes_is_nonempty(self):
        assert len(HealthHandler.ROUTES) > 10

    def test_public_routes_subset_of_routes(self):
        """All public routes should be in the main ROUTES list."""
        for pub in HealthHandler.PUBLIC_ROUTES:
            assert pub in HealthHandler.ROUTES, f"Public route not in ROUTES: {pub}"
