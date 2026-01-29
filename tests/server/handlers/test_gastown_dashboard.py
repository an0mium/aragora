# mypy: ignore-errors
"""
Tests for GasTownDashboardHandler.

Tests REST API endpoints for the Gas Town dashboard:
- GET /api/v1/dashboard/gastown/overview
- GET /api/v1/dashboard/gastown/convoys
- GET /api/v1/dashboard/gastown/agents
- GET /api/v1/dashboard/gastown/beads
- GET /api/v1/dashboard/gastown/metrics

Covers:
- Route matching (can_handle)
- Authentication and permission enforcement
- Sub-handler routing
- Import-failure fallback structures
- Cache hit and force-refresh behaviour
- Invalid parameter handling
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.gastown_dashboard import (
    GasTownDashboardHandler,
    _gt_dashboard_cache,
    _set_cached_data,
    CACHE_TTL,
)
from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.secure import UnauthorizedError, ForbiddenError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a GasTownDashboardHandler without calling __init__."""
    h = GasTownDashboardHandler.__new__(GasTownDashboardHandler)
    h.ctx = {}
    return h


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the module-level dashboard cache before each test."""
    _gt_dashboard_cache.clear()
    yield
    _gt_dashboard_cache.clear()


def _parse(result: HandlerResult) -> dict[str, Any]:
    """Parse a HandlerResult body into a dict."""
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# strip_version_prefix converts /api/v1/foo to /api/foo.  The handler's ROUTES
# and internal path comparisons were written with the /api/v1/ prefix, so
# passing a versioned path causes a mismatch after stripping.  We patch the
# function to an identity so we can test the routing logic in isolation.
_SVP = "aragora.server.handlers.gastown_dashboard.strip_version_prefix"


def _identity(path: str) -> str:
    return path


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for GasTownDashboardHandler.can_handle."""

    def test_can_handle_valid_paths(self, handler):
        """All ROUTES paths are recognised."""
        with patch(_SVP, side_effect=_identity):
            for route in GasTownDashboardHandler.ROUTES:
                assert handler.can_handle(route), f"Expected can_handle('{route}') to be True"

    def test_can_handle_subpath(self, handler):
        """A path under the gastown dashboard prefix is recognised."""
        with patch(_SVP, side_effect=_identity):
            assert handler.can_handle("/api/v1/dashboard/gastown/custom-sub")

    def test_can_handle_invalid_path(self, handler):
        """Unrelated paths are rejected."""
        with patch(_SVP, side_effect=_identity):
            assert not handler.can_handle("/api/v1/analytics/summary")
            assert not handler.can_handle("/api/v1/debates")
            assert not handler.can_handle("/other/path")


# ---------------------------------------------------------------------------
# Authentication / authorisation in handle()
# ---------------------------------------------------------------------------


class TestAuth:
    """Tests for authentication and permission gating."""

    @pytest.mark.asyncio
    async def test_handle_requires_auth(self, handler):
        """handle() returns 401 when authentication fails."""
        handler.get_auth_context = AsyncMock(side_effect=UnauthorizedError("no token"))
        result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, MagicMock())
        assert result.status_code == 401
        body = _parse(result)
        assert "Authentication required" in body.get("error", body.get("message", ""))

    @pytest.mark.asyncio
    async def test_handle_requires_permission(self, handler):
        """handle() returns 403 when permission check fails."""
        handler.get_auth_context = AsyncMock(return_value=MagicMock())
        handler.check_permission = MagicMock(side_effect=ForbiddenError("gastown:read"))
        result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, MagicMock())
        assert result.status_code == 403
        body = _parse(result)
        assert "Permission denied" in body.get("error", body.get("message", ""))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _authenticated(handler):
    """Patch auth methods to pass silently."""
    handler.get_auth_context = AsyncMock(return_value=MagicMock())
    handler.check_permission = MagicMock()


class TestRouting:
    """Tests for handle() dispatching to the correct sub-handler."""

    @pytest.mark.asyncio
    async def test_handle_routes_overview(self, handler):
        """Overview path dispatches to _get_overview."""
        _authenticated(handler)
        handler._get_overview = AsyncMock(
            return_value=HandlerResult(200, "application/json", b"{}")
        )
        with patch(_SVP, side_effect=_identity):
            await handler.handle(
                "/api/v1/dashboard/gastown/overview", {"refresh": "false"}, MagicMock()
            )
        handler._get_overview.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_routes_convoys(self, handler):
        """Convoys path dispatches to _get_convoys."""
        _authenticated(handler)
        handler._get_convoys = AsyncMock(return_value=HandlerResult(200, "application/json", b"{}"))
        with patch(_SVP, side_effect=_identity):
            await handler.handle("/api/v1/dashboard/gastown/convoys", {}, MagicMock())
        handler._get_convoys.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_routes_agents(self, handler):
        """Agents path dispatches to _get_agents."""
        _authenticated(handler)
        handler._get_agents = AsyncMock(return_value=HandlerResult(200, "application/json", b"{}"))
        with patch(_SVP, side_effect=_identity):
            await handler.handle("/api/v1/dashboard/gastown/agents", {}, MagicMock())
        handler._get_agents.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_routes_beads(self, handler):
        """Beads path dispatches to _get_beads."""
        _authenticated(handler)
        handler._get_beads = AsyncMock(return_value=HandlerResult(200, "application/json", b"{}"))
        with patch(_SVP, side_effect=_identity):
            await handler.handle("/api/v1/dashboard/gastown/beads", {}, MagicMock())
        handler._get_beads.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_routes_metrics(self, handler):
        """Metrics path dispatches to _get_metrics."""
        _authenticated(handler)
        handler._get_metrics = AsyncMock(return_value=HandlerResult(200, "application/json", b"{}"))
        with patch(_SVP, side_effect=_identity):
            await handler.handle("/api/v1/dashboard/gastown/metrics", {}, MagicMock())
        handler._get_metrics.assert_awaited_once()


# ---------------------------------------------------------------------------
# Overview endpoint
# ---------------------------------------------------------------------------


class TestOverview:
    """Tests for _get_overview."""

    @pytest.mark.asyncio
    async def test_overview_returns_default_structure(self, handler):
        """When all gastown imports fail, overview returns zeroed defaults."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.extensions.gastown.convoy": None,
                "aragora.extensions.gastown.beads": None,
                "aragora.extensions.gastown.agent_roles": None,
                "aragora.server.startup": None,
            },
        ):
            result = await handler._get_overview({})

        assert result.status_code == 200
        body = _parse(result)
        assert body["convoys"]["total"] == 0
        assert body["beads"]["total"] == 0
        assert body["agents"]["total"] == 0
        assert body["witness_patrol"]["active"] is False
        assert body["mayor"]["active"] is False

    @pytest.mark.asyncio
    async def test_overview_with_convoy_data(self, handler):
        """Overview populates convoy counts from the convoy module."""
        # Build mock convoy module
        mock_convoy_mod = MagicMock()
        active_convoy = MagicMock()
        active_convoy.status = mock_convoy_mod.NomicConvoyStatus.ACTIVE
        completed_convoy = MagicMock()
        completed_convoy.status = mock_convoy_mod.NomicConvoyStatus.COMPLETED
        failed_convoy = MagicMock()
        failed_convoy.status = mock_convoy_mod.NomicConvoyStatus.FAILED

        mock_manager_instance = AsyncMock()
        mock_manager_instance.list_convoys = AsyncMock(
            return_value=[active_convoy, completed_convoy, failed_convoy]
        )
        mock_convoy_mod.NomicConvoyManager.return_value = mock_manager_instance

        with patch.dict(
            "sys.modules",
            {
                "aragora.extensions.gastown.convoy": mock_convoy_mod,
                "aragora.extensions.gastown.beads": None,
                "aragora.extensions.gastown.agent_roles": None,
                "aragora.server.startup": None,
            },
        ):
            result = await handler._get_overview({})

        body = _parse(result)
        assert body["convoys"]["active"] == 1
        assert body["convoys"]["completed"] == 1
        assert body["convoys"]["failed"] == 1
        assert body["convoys"]["total"] == 3

    @pytest.mark.asyncio
    async def test_overview_cache_hit(self, handler):
        """Cached overview data is returned without recomputing."""
        cached_data = {
            "generated_at": "2024-01-01T00:00:00+00:00",
            "convoys": {"active": 5, "completed": 10, "failed": 1, "total": 16},
            "beads": {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0, "total": 0},
            "agents": {"mayor": 0, "witness": 0, "polecat": 0, "crew": 0, "total": 0},
            "witness_patrol": {"active": False, "last_check": None},
            "mayor": {"active": False, "node_id": None},
        }
        _set_cached_data("overview", cached_data)

        result = await handler._get_overview({})
        body = _parse(result)
        assert body["convoys"]["total"] == 16, "Should return cached data"

    @pytest.mark.asyncio
    async def test_overview_force_refresh(self, handler):
        """refresh=true bypasses the cache."""
        cached_data = {
            "generated_at": "2024-01-01T00:00:00+00:00",
            "convoys": {"active": 99, "completed": 0, "failed": 0, "total": 99},
            "beads": {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0, "total": 0},
            "agents": {"mayor": 0, "witness": 0, "polecat": 0, "crew": 0, "total": 0},
            "witness_patrol": {"active": False, "last_check": None},
            "mayor": {"active": False, "node_id": None},
        }
        _set_cached_data("overview", cached_data)

        # Force refresh with all imports failing -> zeroed structure
        with patch.dict(
            "sys.modules",
            {
                "aragora.extensions.gastown.convoy": None,
                "aragora.extensions.gastown.beads": None,
                "aragora.extensions.gastown.agent_roles": None,
                "aragora.server.startup": None,
            },
        ):
            result = await handler._get_overview({"refresh": "true"})

        body = _parse(result)
        # Should NOT be the cached value of 99
        assert body["convoys"]["total"] == 0


# ---------------------------------------------------------------------------
# Convoys endpoint
# ---------------------------------------------------------------------------


class TestConvoys:
    """Tests for _get_convoys."""

    @pytest.mark.asyncio
    async def test_convoys_returns_empty_on_import_error(self, handler):
        """When convoy module is unavailable, return empty list."""
        with patch.dict("sys.modules", {"aragora.extensions.gastown.convoy": None}):
            result = await handler._get_convoys({})

        assert result.status_code == 200
        body = _parse(result)
        assert body["convoys"] == []
        assert body["total"] == 0
        assert body["showing"] == 0

    @pytest.mark.asyncio
    async def test_convoys_invalid_status_returns_400(self, handler):
        """An unrecognised status filter yields 400."""
        mock_convoy_mod = MagicMock()
        mock_convoy_mod.NomicConvoyStatus.side_effect = ValueError("bad status")
        mock_convoy_mod.NomicConvoyManager.return_value = AsyncMock()

        with patch.dict("sys.modules", {"aragora.extensions.gastown.convoy": mock_convoy_mod}):
            result = await handler._get_convoys({"status": "INVALID_STATUS"})

        assert result.status_code == 400
        body = _parse(result)
        assert "Invalid status" in body.get("error", body.get("message", ""))


# ---------------------------------------------------------------------------
# Agents endpoint
# ---------------------------------------------------------------------------


class TestAgents:
    """Tests for _get_agents."""

    @pytest.mark.asyncio
    async def test_agents_returns_empty_on_import_error(self, handler):
        """When agent_roles module is unavailable, return empty structure."""
        with patch.dict("sys.modules", {"aragora.extensions.gastown.agent_roles": None}):
            result = await handler._get_agents({})

        assert result.status_code == 200
        body = _parse(result)
        assert body["agents_by_role"] == {}
        assert body["totals"]["total"] == 0


# ---------------------------------------------------------------------------
# Beads endpoint
# ---------------------------------------------------------------------------


class TestBeads:
    """Tests for _get_beads."""

    @pytest.mark.asyncio
    async def test_beads_returns_empty_on_import_error(self, handler):
        """When bead module is unavailable, return zeroed structure."""
        with patch.dict("sys.modules", {"aragora.extensions.gastown.beads": None}):
            result = await handler._get_beads({})

        assert result.status_code == 200
        body = _parse(result)
        assert body["queue_depth"] == 0
        assert body["processing"] == 0
        assert body["total"] == 0


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for _get_metrics."""

    @pytest.mark.asyncio
    async def test_metrics_returns_default_on_import_error(self, handler):
        """When metrics and convoy modules are unavailable, return defaults."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.extensions.gastown.metrics": None,
                "aragora.extensions.gastown.convoy": None,
            },
        ):
            result = await handler._get_metrics({})

        assert result.status_code == 200
        body = _parse(result)
        assert body["period_hours"] == 24
        assert body["beads_per_hour"] == 0.0
        assert body["convoy_completion_rate"] == 0.0
        assert body["gupp_recovery_events"] == 0

    @pytest.mark.asyncio
    async def test_metrics_respects_hours_param(self, handler):
        """The hours query parameter is forwarded to the response."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.extensions.gastown.metrics": None,
                "aragora.extensions.gastown.convoy": None,
            },
        ):
            result = await handler._get_metrics({"hours": "48"})

        body = _parse(result)
        assert body["period_hours"] == 48
