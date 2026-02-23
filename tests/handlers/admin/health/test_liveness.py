"""Comprehensive tests for LivenessHandler.

Tests the LivenessHandler class in aragora/server/handlers/admin/health/liveness.py:

  TestLivenessHandlerInit          - Constructor and ctx initialization
  TestLivenessHandlerRoutes        - ROUTES and PUBLIC_ROUTES class attributes
  TestLivenessHandlerCanHandle     - can_handle() routing logic
  TestLivenessHandlerHandle        - handle() async dispatch
  TestLivenessHandlerLivenessProbe - _liveness_probe() delegation
  TestLivenessHandlerHealthy       - Full path: healthy server
  TestLivenessHandlerDegraded      - Full path: degraded server
  TestLivenessHandlerImportError   - Full path: degraded_mode unavailable
  TestLivenessHandlerEdgeCases     - Edge cases and integration

30+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.admin.health.liveness import LivenessHandler


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


def _make_degraded_module(
    is_degraded_val: bool = False,
    degraded_reason: str = "",
):
    """Create a fake aragora.server.degraded_mode module."""
    mod = types.ModuleType("aragora.server.degraded_mode")
    mod.is_degraded = lambda: is_degraded_val
    mod.get_degraded_reason = lambda: degraded_reason
    return mod


def _patch_degraded(is_degraded_val=False, reason=""):
    """Patch degraded_mode module in sys.modules."""
    mod = _make_degraded_module(is_degraded_val, reason)
    return patch.dict(sys.modules, {"aragora.server.degraded_mode": mod})


def _remove_degraded():
    """Remove degraded_mode so ImportError is raised."""
    return patch.dict(sys.modules, {"aragora.server.degraded_mode": None})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_health_cache():
    """Clear the health cache before each test to avoid cross-test pollution."""
    import aragora.server.handlers.admin.health as pkg

    pkg._HEALTH_CACHE.clear()
    pkg._HEALTH_CACHE_TIMESTAMPS.clear()
    yield
    pkg._HEALTH_CACHE.clear()
    pkg._HEALTH_CACHE_TIMESTAMPS.clear()


@pytest.fixture
def handler():
    """Create a LivenessHandler with empty context."""
    return LivenessHandler()


@pytest.fixture
def handler_with_ctx():
    """Create a LivenessHandler with a populated context."""
    return LivenessHandler(ctx={"storage": MagicMock(), "elo_system": MagicMock()})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler object."""
    h = MagicMock()
    h.rfile = MagicMock()
    h.rfile.read.return_value = b"{}"
    h.headers = {"Content-Length": "2"}
    return h


# ===========================================================================
# TestLivenessHandlerInit
# ===========================================================================


class TestLivenessHandlerInit:
    """Test LivenessHandler constructor and initialization."""

    def test_default_ctx_is_empty_dict(self):
        h = LivenessHandler()
        assert h.ctx == {}

    def test_none_ctx_becomes_empty_dict(self):
        h = LivenessHandler(ctx=None)
        assert h.ctx == {}

    def test_ctx_preserved_when_provided(self):
        ctx = {"storage": "fake_storage", "key": 42}
        h = LivenessHandler(ctx=ctx)
        assert h.ctx is ctx
        assert h.ctx["storage"] == "fake_storage"
        assert h.ctx["key"] == 42

    def test_ctx_with_empty_dict_creates_new_dict(self):
        """Empty dict is falsy, so `ctx or {}` creates a new empty dict."""
        ctx = {}
        h = LivenessHandler(ctx=ctx)
        assert h.ctx == {}
        # Because empty dict is falsy, `ctx or {}` creates a fresh dict
        assert h.ctx is not ctx


# ===========================================================================
# TestLivenessHandlerRoutes
# ===========================================================================


class TestLivenessHandlerRoutes:
    """Test ROUTES and PUBLIC_ROUTES class attributes."""

    def test_routes_contains_healthz(self):
        assert "/healthz" in LivenessHandler.ROUTES

    def test_routes_is_list(self):
        assert isinstance(LivenessHandler.ROUTES, list)

    def test_routes_has_exactly_one_entry(self):
        assert len(LivenessHandler.ROUTES) == 1

    def test_public_routes_contains_healthz(self):
        assert "/healthz" in LivenessHandler.PUBLIC_ROUTES

    def test_public_routes_is_set(self):
        assert isinstance(LivenessHandler.PUBLIC_ROUTES, set)

    def test_public_routes_has_exactly_one_entry(self):
        assert len(LivenessHandler.PUBLIC_ROUTES) == 1

    def test_public_routes_matches_routes(self):
        """All routes should be public (no auth required for liveness)."""
        assert set(LivenessHandler.ROUTES) == LivenessHandler.PUBLIC_ROUTES


# ===========================================================================
# TestLivenessHandlerCanHandle
# ===========================================================================


class TestLivenessHandlerCanHandle:
    """Test can_handle() routing logic."""

    def test_can_handle_healthz(self, handler):
        assert handler.can_handle("/healthz") is True

    def test_cannot_handle_readyz(self, handler):
        assert handler.can_handle("/readyz") is False

    def test_cannot_handle_api_health(self, handler):
        assert handler.can_handle("/api/health") is False

    def test_cannot_handle_empty_string(self, handler):
        assert handler.can_handle("") is False

    def test_cannot_handle_slash(self, handler):
        assert handler.can_handle("/") is False

    def test_cannot_handle_healthz_trailing_slash(self, handler):
        assert handler.can_handle("/healthz/") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/health") is False

    def test_cannot_handle_case_sensitive(self, handler):
        assert handler.can_handle("/HEALTHZ") is False

    def test_cannot_handle_healthz_with_query(self, handler):
        """Query strings are not part of the path."""
        assert handler.can_handle("/healthz?verbose=true") is False


# ===========================================================================
# TestLivenessHandlerHandle
# ===========================================================================


class TestLivenessHandlerHandle:
    """Test handle() async dispatch."""

    @pytest.mark.asyncio
    async def test_handle_healthz_returns_result(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert result is not None
        assert _status(result) == 200
        assert _body(result)["status"] == "ok"

    @pytest.mark.asyncio
    async def test_handle_unknown_path_returns_none(self, handler, mock_http_handler):
        result = await handler.handle("/unknown", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_readyz_returns_none(self, handler, mock_http_handler):
        result = await handler.handle("/readyz", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_empty_path_returns_none(self, handler, mock_http_handler):
        result = await handler.handle("", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_ignores_query_params(self, handler, mock_http_handler):
        """Query params are passed but ignored for liveness."""
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {"verbose": "true"}, mock_http_handler)
        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_ignores_handler_arg(self, handler):
        """The HTTP handler is passed through but not used by liveness."""
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {}, None)
        assert result is not None
        assert _status(result) == 200


# ===========================================================================
# TestLivenessHandlerLivenessProbe
# ===========================================================================


class TestLivenessHandlerLivenessProbe:
    """Test _liveness_probe() delegation to kubernetes.liveness_probe."""

    def test_delegates_to_kubernetes_liveness_probe(self, handler):
        mock_result = MagicMock()
        with patch(
            "aragora.server.handlers.admin.health.liveness.liveness_probe",
            return_value=mock_result,
        ) as mock_probe:
            result = handler._liveness_probe()
        mock_probe.assert_called_once_with(handler)
        assert result is mock_result

    def test_passes_self_as_handler_arg(self):
        h = LivenessHandler(ctx={"key": "value"})
        with patch(
            "aragora.server.handlers.admin.health.liveness.liveness_probe",
        ) as mock_probe:
            h._liveness_probe()
        args, _ = mock_probe.call_args
        assert args[0] is h


# ===========================================================================
# TestLivenessHandlerHealthy
# ===========================================================================


class TestLivenessHandlerHealthy:
    """Full path tests: healthy server returns 200 with status ok."""

    @pytest.mark.asyncio
    async def test_healthy_returns_200(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_healthy_returns_status_ok(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _body(result)["status"] == "ok"

    @pytest.mark.asyncio
    async def test_healthy_has_no_degraded_key(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert "degraded" not in _body(result)

    @pytest.mark.asyncio
    async def test_healthy_has_no_note_key(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert "note" not in _body(result)

    @pytest.mark.asyncio
    async def test_healthy_has_no_degraded_reason_key(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert "degraded_reason" not in _body(result)

    @pytest.mark.asyncio
    async def test_healthy_response_is_json(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=False):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert result.content_type == "application/json"


# ===========================================================================
# TestLivenessHandlerDegraded
# ===========================================================================


class TestLivenessHandlerDegraded:
    """Full path tests: degraded server still returns 200 (alive)."""

    @pytest.mark.asyncio
    async def test_degraded_still_returns_200(self, handler, mock_http_handler):
        """Container should NOT be restarted for degraded mode."""
        with _patch_degraded(is_degraded_val=True, reason="Missing API key"):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_degraded_body_marks_degraded_true(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=True, reason="Missing API key"):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _body(result)["degraded"] is True

    @pytest.mark.asyncio
    async def test_degraded_body_has_status_ok(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=True, reason="Missing API key"):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _body(result)["status"] == "ok"

    @pytest.mark.asyncio
    async def test_degraded_body_has_reason(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=True, reason="Missing API key"):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _body(result)["degraded_reason"] == "Missing API key"

    @pytest.mark.asyncio
    async def test_degraded_body_has_note(self, handler, mock_http_handler):
        with _patch_degraded(is_degraded_val=True, reason="reason"):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert "Check /api/health" in _body(result)["note"]

    @pytest.mark.asyncio
    async def test_degraded_reason_truncated_to_100(self, handler, mock_http_handler):
        long_reason = "x" * 200
        with _patch_degraded(is_degraded_val=True, reason=long_reason):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert len(_body(result)["degraded_reason"]) == 100


# ===========================================================================
# TestLivenessHandlerImportError
# ===========================================================================


class TestLivenessHandlerImportError:
    """Full path tests: when degraded_mode module is unavailable."""

    @pytest.mark.asyncio
    async def test_returns_200_on_import_error(self, handler, mock_http_handler):
        with _remove_degraded():
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_returns_status_ok_on_import_error(self, handler, mock_http_handler):
        with _remove_degraded():
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _body(result)["status"] == "ok"

    @pytest.mark.asyncio
    async def test_no_degraded_fields_on_import_error(self, handler, mock_http_handler):
        with _remove_degraded():
            result = await handler.handle("/healthz", {}, mock_http_handler)
        body = _body(result)
        assert "degraded" not in body
        assert "degraded_reason" not in body
        assert "note" not in body


# ===========================================================================
# TestLivenessHandlerEdgeCases
# ===========================================================================


class TestLivenessHandlerEdgeCases:
    """Edge cases and integration tests."""

    def test_module_exports_liveness_handler(self):
        """LivenessHandler should be in __all__."""
        from aragora.server.handlers.admin.health import liveness

        assert "LivenessHandler" in liveness.__all__

    def test_handler_inherits_from_secure_handler(self):
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(LivenessHandler, SecureHandler)

    @pytest.mark.asyncio
    async def test_with_populated_context(self, handler_with_ctx, mock_http_handler):
        """Handler with context still works for liveness."""
        with _patch_degraded(is_degraded_val=False):
            result = await handler_with_ctx.handle("/healthz", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["status"] == "ok"

    @pytest.mark.asyncio
    async def test_multiple_calls_are_idempotent(self, handler, mock_http_handler):
        """Repeated calls should return identical results."""
        with _patch_degraded(is_degraded_val=False):
            result1 = await handler.handle("/healthz", {}, mock_http_handler)
            result2 = await handler.handle("/healthz", {}, mock_http_handler)
        assert _body(result1) == _body(result2)
        assert _status(result1) == _status(result2)

    def test_handler_is_importable_from_health_package(self):
        """LivenessHandler should be importable from the health package."""
        from aragora.server.handlers.admin.health import LivenessHandler as LH

        assert LH is LivenessHandler

    @pytest.mark.asyncio
    async def test_degraded_empty_reason_string(self, handler, mock_http_handler):
        """Empty degraded reason should still work."""
        with _patch_degraded(is_degraded_val=True, reason=""):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["degraded"] is True
        assert _body(result)["degraded_reason"] == ""

    @pytest.mark.asyncio
    async def test_degraded_reason_exactly_100_chars(self, handler, mock_http_handler):
        """Reason string of exactly 100 chars should not be truncated."""
        reason_100 = "a" * 100
        with _patch_degraded(is_degraded_val=True, reason=reason_100):
            result = await handler.handle("/healthz", {}, mock_http_handler)
        assert len(_body(result)["degraded_reason"]) == 100
        assert _body(result)["degraded_reason"] == reason_100
