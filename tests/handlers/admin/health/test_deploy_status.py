"""Tests for DeployStatusHandler.

Covers all routes and behaviour of the DeployStatusHandler class:
- GET /api/deploy/status     - Deploy status (non-v1)
- GET /api/v1/deploy/status  - Deploy status (v1)

Tests authentication enforcement, health/readiness checks, uptime calculation,
and build info display.
"""

from __future__ import annotations

import json
import time
import types
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.server.handlers.admin.health.deploy_status import DeployStatusHandler


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
# Mock objects
# ---------------------------------------------------------------------------


MOCK_BUILD_INFO = {
    "sha": "abc123def456789012345678901234567890abcd",
    "sha_short": "abc123d",
    "build_time": "2026-02-24T08:00:00Z",
    "version": "2.5.0",
}


def _make_build_info_module(build_info: dict | None = None):
    """Create a fake aragora.server.build_info module."""
    mod = types.ModuleType("aragora.server.build_info")
    info = build_info or MOCK_BUILD_INFO
    mod.get_build_info = lambda: info
    return mod


def _make_degraded_module(is_degraded_val: bool = False):
    """Create a fake aragora.server.degraded_mode module."""
    mod = types.ModuleType("aragora.server.degraded_mode")
    mod.is_degraded = lambda: is_degraded_val
    return mod


def _make_server_module(is_ready: bool = True):
    """Create a fake aragora.server.unified_server module."""
    mod = types.ModuleType("aragora.server.unified_server")
    mod.is_server_ready = lambda: is_ready
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a DeployStatusHandler instance."""
    return DeployStatusHandler(ctx={})


@pytest.fixture
def mock_start_time():
    """Patch _SERVER_START_TIME to a known value."""
    start = time.time() - 3600  # 1 hour ago
    with patch(
        "aragora.server.handlers.admin.health._SERVER_START_TIME",
        start,
    ):
        yield start


@pytest.fixture
def mock_deps(mock_start_time):
    """Patch all dependencies for deploy_status."""
    with (
        patch(
            "aragora.server.build_info.get_build_info",
            create=True,
        ) as mock_build,
        patch.dict(
            "sys.modules",
            {
                "aragora.server.build_info": _make_build_info_module(),
                "aragora.server.degraded_mode": _make_degraded_module(False),
                "aragora.server.unified_server": _make_server_module(True),
            },
        ),
    ):
        # We need to patch the import inside _deploy_status
        yield {
            "start_time": mock_start_time,
        }


# ---------------------------------------------------------------------------
# ROUTES and can_handle
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES class attribute and can_handle."""

    def test_routes_contains_both_endpoints(self):
        expected = [
            "/api/deploy/status",
            "/api/v1/deploy/status",
        ]
        for route in expected:
            assert route in DeployStatusHandler.ROUTES, f"Missing route: {route}"

    def test_can_handle_v1_path(self, handler):
        assert handler.can_handle("/api/v1/deploy/status")

    def test_can_handle_non_v1_path(self, handler):
        assert handler.can_handle("/api/deploy/status")

    def test_can_handle_rejects_unknown_paths(self, handler):
        assert not handler.can_handle("/api/v1/deploy/other")
        assert not handler.can_handle("/api/v1/health")
        assert not handler.can_handle("/api/deploy")


# ---------------------------------------------------------------------------
# handle() - authentication
# ---------------------------------------------------------------------------


class TestHandleAuthentication:
    """Test authentication enforcement in handle()."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_unauthenticated_returns_401(self):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        handler = DeployStatusHandler(ctx={})

        async def raise_unauth(self, request, require_auth=True):
            raise UnauthorizedError("No token")

        with patch.object(
            DeployStatusHandler,
            "get_auth_context",
            raise_unauth,
        ):
            result = await handler.handle("/api/v1/deploy/status", {}, MagicMock())

        assert _status(result) == 401
        body = _body(result)
        assert "authentication" in json.dumps(body).lower() or "auth" in json.dumps(body).lower()

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_forbidden_returns_403(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        handler = DeployStatusHandler(ctx={})

        async def raise_forbidden(self, request, require_auth=True):
            return MagicMock()  # Return auth context

        def deny_permission(self, auth_ctx, perm, resource_id=None):
            raise ForbiddenError("Denied")

        with (
            patch.object(
                DeployStatusHandler,
                "get_auth_context",
                raise_forbidden,
            ),
            patch.object(
                DeployStatusHandler,
                "check_permission",
                deny_permission,
            ),
        ):
            result = await handler.handle("/api/v1/deploy/status", {}, MagicMock())

        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_handle_non_matching_path_returns_none(self, handler):
        result = await handler.handle("/api/v1/other", {}, MagicMock())

        assert result is None


# ---------------------------------------------------------------------------
# _deploy_status() - core response
# ---------------------------------------------------------------------------


class TestDeployStatus:
    """Test the _deploy_status method."""

    def test_deploy_status_returns_all_sections(self, handler):
        start = time.time() - 120
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        assert "deploy" in body
        assert "health" in body
        assert "uptime" in body
        assert "timestamp" in body

    def test_deploy_section_has_sha(self, handler):
        start = time.time() - 60
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        deploy = body["deploy"]
        assert deploy["sha"] == MOCK_BUILD_INFO["sha"]
        assert deploy["sha_short"] == MOCK_BUILD_INFO["sha_short"]
        assert deploy["build_time"] == MOCK_BUILD_INFO["build_time"]
        assert deploy["version"] == MOCK_BUILD_INFO["version"]

    def test_health_section_healthy(self, handler):
        start = time.time() - 60
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        health = body["health"]
        assert health["backend"] == "healthy"
        assert health["server_ready"] is True

    def test_health_section_degraded(self, handler):
        start = time.time() - 60
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(True),
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        assert body["health"]["backend"] == "degraded"

    def test_health_section_server_not_ready(self, handler):
        start = time.time() - 60
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": _make_server_module(False),
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        assert body["health"]["server_ready"] is False

    def test_uptime_section(self, handler):
        start = time.time() - 7200  # 2 hours ago
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        uptime = body["uptime"]
        assert "seconds" in uptime
        assert "started_at" in uptime
        # Uptime should be approximately 7200 seconds (within margin)
        assert uptime["seconds"] >= 7100
        assert uptime["seconds"] <= 7300

    def test_timestamp_is_utc(self, handler):
        start = time.time() - 60
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        assert body["timestamp"].endswith("Z")


# ---------------------------------------------------------------------------
# _deploy_status() - import error resilience
# ---------------------------------------------------------------------------


class TestImportErrorResilience:
    """Test graceful degradation when optional imports are unavailable."""

    def test_degraded_mode_import_error_defaults_healthy(self, handler):
        start = time.time() - 60

        # Remove degraded_mode from sys.modules to trigger ImportError
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": None,
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        # When degraded_mode import fails, defaults to healthy
        assert body["health"]["backend"] == "healthy"

    def test_unified_server_import_error_defaults_ready(self, handler):
        start = time.time() - 60

        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": None,
                },
            ),
        ):
            result = handler._deploy_status()

        body = _body(result)
        # When unified_server import fails, defaults to ready
        assert body["health"]["server_ready"] is True


# ---------------------------------------------------------------------------
# handle() - full flow
# ---------------------------------------------------------------------------


class TestHandleFullFlow:
    """Test the complete handle() method for authorized requests."""

    @pytest.mark.asyncio
    async def test_handle_v1_path(self, handler):
        start = time.time() - 60
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = await handler.handle("/api/v1/deploy/status", {}, MagicMock())

        body = _body(result)
        assert "deploy" in body
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_non_v1_path(self, handler):
        start = time.time() - 60
        with (
            patch(
                "aragora.server.handlers.admin.health._SERVER_START_TIME",
                start,
            ),
            patch(
                "aragora.server.build_info.get_build_info",
                return_value=MOCK_BUILD_INFO,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": _make_degraded_module(False),
                    "aragora.server.unified_server": _make_server_module(True),
                },
            ),
        ):
            result = await handler.handle("/api/deploy/status", {}, MagicMock())

        body = _body(result)
        assert "deploy" in body
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# DEPLOY_PERMISSION
# ---------------------------------------------------------------------------


class TestPermission:
    """Test the DEPLOY_PERMISSION class attribute."""

    def test_deploy_permission_is_system_health_read(self):
        assert DeployStatusHandler.DEPLOY_PERMISSION == "system.health.read"


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    """Test handler initialization."""

    def test_default_ctx(self):
        handler = DeployStatusHandler()
        assert handler.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "value"}
        handler = DeployStatusHandler(ctx=ctx)
        assert handler.ctx == ctx
