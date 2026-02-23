"""Comprehensive tests for detailed health check handlers.

Tests the four public functions in aragora/server/handlers/admin/health/detailed.py:

  TestHealthCheck                - health_check() comprehensive health check
  TestWebsocketHealth            - websocket_health() basic WS health
  TestDetailedHealthCheck        - detailed_health_check() with observer metrics
  TestDeepHealthCheck            - deep_health_check() full dependency verification

80+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.admin.health import HealthHandler


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


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            }
        else:
            self.rfile.read.return_value = b""
            self.headers = {
                "Content-Length": "0",
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            }
        self.client_address = ("127.0.0.1", 12345)


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
def mock_http():
    """Default mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture
def tmp_nomic_dir(tmp_path):
    """Temporary directory that exists, usable as nomic_dir."""
    d = tmp_path / "nomic"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Common patch targets
# ---------------------------------------------------------------------------

# Top-level imports in detailed.py (imported at module load via __init__.py)
_P_FS = "aragora.server.handlers.admin.health.detailed.check_filesystem_health"
_P_REDIS = "aragora.server.handlers.admin.health.detailed.check_redis_health"
_P_AI = "aragora.server.handlers.admin.health.detailed.check_ai_providers_health"
_P_SEC = "aragora.server.handlers.admin.health.detailed.check_security_services"

# Locally imported inside deep_health_check from ..health_utils
_P_STRIPE = "aragora.server.handlers.admin.health_utils.check_stripe_health"
_P_SLACK = "aragora.server.handlers.admin.health_utils.check_slack_health"

# Healthy defaults
_HEALTHY_FS = {"healthy": True, "path": "/tmp"}
_HEALTHY_REDIS = {"healthy": True, "configured": False, "note": "Redis not configured"}
_HEALTHY_AI = {"healthy": True, "any_available": True, "available_count": 1, "providers": {}}
_HEALTHY_SECURITY = {"healthy": True, "encryption_configured": True}
_HEALTHY_STRIPE = {"healthy": True, "configured": False}
_HEALTHY_SLACK = {"healthy": True, "configured": False}


# ============================================================================
# TestHealthCheck - health_check() function
# ============================================================================


class TestHealthCheck:
    """Tests for health_check() - comprehensive health endpoint."""

    def _call(self, handler):
        """Invoke health_check on the handler."""
        return handler._health_check()

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_healthy_status(self, _sec, _ai, _redis, _fs):
        """All checks pass yields status=healthy and 200."""
        h = _make_handler()
        result = self._call(h)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_response_has_required_fields(self, _sec, _ai, _redis, _fs):
        """Response contains all required top-level fields."""
        h = _make_handler()
        body = _body(self._call(h))
        for key in (
            "status",
            "version",
            "uptime_seconds",
            "checks",
            "timestamp",
            "response_time_ms",
            "demo_mode",
        ):
            assert key in body, f"Missing key: {key}"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_false_by_default(self, _sec, _ai, _redis, _fs):
        """Demo mode is False when env var not set."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["demo_mode"] is False

    @patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_true_when_set(self, _sec, _ai, _redis, _fs):
        """Demo mode detects ARAGORA_DEMO_MODE=true."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["demo_mode"] is True

    @patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "1"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_accepts_1(self, _sec, _ai, _redis, _fs):
        """Demo mode detects ARAGORA_DEMO_MODE=1."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["demo_mode"] is True

    @patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "yes"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_accepts_yes(self, _sec, _ai, _redis, _fs):
        """Demo mode detects ARAGORA_DEMO_MODE=yes."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["demo_mode"] is True

    # -- Degraded mode checks --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_degraded_mode_not_degraded(self, _sec, _ai, _redis, _fs):
        """Non-degraded server shows healthy degraded_mode check."""
        h = _make_handler()
        body = _body(self._call(h))
        dm = body["checks"]["degraded_mode"]
        assert dm["healthy"] is True

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_degraded_mode_is_degraded(self, _sec, _ai, _redis, _fs):
        """When server is degraded, check marks unhealthy and overall = 503."""
        mock_state = MagicMock()
        mock_state.reason = "startup failure"
        mock_state.error_code.value = "STARTUP_ERROR"
        mock_state.recovery_hint = "restart"
        mock_state.timestamp = "2026-01-01T00:00:00Z"

        mock_degraded_mod = MagicMock()
        mock_degraded_mod.is_degraded.return_value = True
        mock_degraded_mod.get_degraded_state.return_value = mock_state

        with patch.dict("sys.modules", {"aragora.server.degraded_mode": mock_degraded_mod}):
            h = _make_handler()
            result = self._call(h)
            body = _body(result)
            assert _status(result) == 503
            assert body["status"] == "degraded"
            assert body["checks"]["degraded_mode"]["healthy"] is False
            assert body["checks"]["degraded_mode"]["reason"] == "startup failure"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_degraded_mode_import_error(self, _sec, _ai, _redis, _fs):
        """degraded_mode module not available -> healthy with module_not_available."""
        with patch.dict("sys.modules", {"aragora.server.degraded_mode": None}):
            h = _make_handler()
            body = _body(self._call(h))
            dm = body["checks"]["degraded_mode"]
            assert dm["healthy"] is True
            assert dm["status"] == "module_not_available"

    # -- Database check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_database_healthy(self, _sec, _ai, _redis, _fs):
        """Database check passes when storage is available."""
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = []
        h = _make_handler({"storage": mock_storage})
        body = _body(self._call(h))
        assert body["checks"]["database"]["healthy"] is True
        assert "latency_ms" in body["checks"]["database"]

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_database_not_initialized(self, _sec, _ai, _redis, _fs):
        """No storage -> warning but still healthy."""
        h = _make_handler()
        body = _body(self._call(h))
        db = body["checks"]["database"]
        assert db["healthy"] is True
        assert "warning" in db
        assert db["initialized"] is False

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_database_error_is_non_critical(self, _sec, _ai, _redis, _fs):
        """Database error is downgraded to warning (healthy=True)."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = RuntimeError("db down")
        h = _make_handler({"storage": mock_storage})
        body = _body(self._call(h))
        db = body["checks"]["database"]
        assert db["healthy"] is True
        assert db["initialized"] is False

    # -- ELO system check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_elo_system_healthy(self, _sec, _ai, _redis, _fs):
        """ELO system check passes when available."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        h = _make_handler({"elo_system": mock_elo})
        body = _body(self._call(h))
        assert body["checks"]["elo_system"]["healthy"] is True

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_elo_system_not_initialized(self, _sec, _ai, _redis, _fs):
        """No ELO system -> warning but healthy."""
        h = _make_handler()
        body = _body(self._call(h))
        elo = body["checks"]["elo_system"]
        assert elo["healthy"] is True
        assert "warning" in elo

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_elo_system_error_non_critical(self, _sec, _ai, _redis, _fs):
        """ELO error is downgraded to warning."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = ValueError("elo corrupt")
        h = _make_handler({"elo_system": mock_elo})
        body = _body(self._call(h))
        assert body["checks"]["elo_system"]["healthy"] is True
        assert body["checks"]["elo_system"]["initialized"] is False

    # -- Nomic directory check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_nomic_dir_exists(self, _sec, _ai, _redis, _fs, tmp_nomic_dir):
        """Nomic dir exists -> healthy with path."""
        h = _make_handler({"nomic_dir": tmp_nomic_dir})
        body = _body(self._call(h))
        assert body["checks"]["nomic_dir"]["healthy"] is True
        assert body["checks"]["nomic_dir"]["path"] == str(tmp_nomic_dir)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_nomic_dir_not_configured(self, _sec, _ai, _redis, _fs):
        """No nomic dir -> downgraded to warning."""
        h = _make_handler()
        body = _body(self._call(h))
        nd = body["checks"]["nomic_dir"]
        assert nd["healthy"] is True
        assert "warning" in nd

    # -- Filesystem check --

    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_filesystem_unhealthy_causes_503(self, _sec, _ai, _redis):
        """Filesystem failure makes overall status degraded."""
        with patch(_P_FS, return_value={"healthy": False, "error": "Permission denied"}):
            h = _make_handler()
            result = self._call(h)
            assert _status(result) == 503
            assert _body(result)["status"] == "degraded"

    # -- Redis check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_redis_configured_and_failing_causes_503(self, _sec, _ai, _fs):
        """Redis configured but failing makes health degraded."""
        with patch(
            _P_REDIS,
            return_value={"healthy": False, "configured": True, "error": "Connection failed"},
        ):
            h = _make_handler()
            result = self._call(h)
            assert _status(result) == 503

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_redis_not_configured_is_ok(self, _sec, _ai, _fs):
        """Redis not configured does not fail health."""
        with patch(_P_REDIS, return_value={"healthy": True, "configured": False}):
            h = _make_handler()
            result = self._call(h)
            assert _status(result) == 200

    # -- AI providers check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_no_ai_providers_adds_warning(self, _sec, _redis, _fs):
        """No AI providers -> warning added but still healthy."""
        with patch(
            _P_AI, return_value={"healthy": True, "any_available": False, "available_count": 0}
        ):
            h = _make_handler()
            result = self._call(h)
            body = _body(result)
            assert _status(result) == 200
            assert "warning" in body["checks"]["ai_providers"]

    # -- WebSocket check in health_check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_websocket_with_manager(self, _sec, _ai, _redis, _fs):
        """WebSocket manager present -> reports client count."""
        ws = MagicMock()
        ws.clients = ["c1", "c2"]
        h = _make_handler({"ws_manager": ws})
        body = _body(self._call(h))
        ws_check = body["checks"]["websocket"]
        assert ws_check["healthy"] is True
        assert ws_check["active_clients"] == 2

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_websocket_no_manager(self, _sec, _ai, _redis, _fs):
        """No ws_manager -> note about separate aiohttp server."""
        h = _make_handler()
        body = _body(self._call(h))
        ws_check = body["checks"]["websocket"]
        assert ws_check["healthy"] is True
        assert "note" in ws_check

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_websocket_manager_type_error(self, _sec, _ai, _redis, _fs):
        """ws_manager.clients raises TypeError -> unhealthy websocket check."""
        # getattr(ws, "clients", []) succeeds, but len() on result raises TypeError
        ws = MagicMock(spec=[])  # empty spec so getattr uses default
        # Override __getattr__ to return an un-lenable object
        ws.clients = MagicMock()
        ws.clients.__len__ = MagicMock(side_effect=TypeError("bad len"))
        h = _make_handler({"ws_manager": ws})
        body = _body(self._call(h))
        assert body["checks"]["websocket"]["healthy"] is False

    # -- Circuit breaker check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_circuit_breakers_healthy(self, _sec, _ai, _redis, _fs):
        """Circuit breakers with <3 open -> healthy."""
        mock_mod = MagicMock()
        mock_mod.get_circuit_breaker_metrics.return_value = {
            "summary": {"open_count": 1, "half_open_count": 0, "closed_count": 5}
        }
        with patch.dict("sys.modules", {"aragora.resilience": mock_mod}):
            h = _make_handler()
            body = _body(self._call(h))
            cb = body["checks"]["circuit_breakers"]
            assert cb["healthy"] is True
            assert cb["open"] == 1

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_circuit_breakers_many_open_503(self, _sec, _ai, _redis, _fs):
        """>=3 open circuit breakers -> degraded."""
        mock_mod = MagicMock()
        mock_mod.get_circuit_breaker_metrics.return_value = {
            "summary": {"open_count": 5, "half_open_count": 0, "closed_count": 1}
        }
        with patch.dict("sys.modules", {"aragora.resilience": mock_mod}):
            h = _make_handler()
            result = self._call(h)
            assert _status(result) == 503

    # -- Rate limiter check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_rate_limiter_stats(self, _sec, _ai, _redis, _fs):
        """Rate limiter stats are included when available."""
        mock_auth_config = MagicMock()
        mock_auth_config.get_rate_limit_stats.return_value = {
            "ip_entries": 10,
            "token_entries": 5,
            "revoked_tokens": 2,
        }
        mock_mod = MagicMock()
        mock_mod.auth_config = mock_auth_config
        with patch.dict("sys.modules", {"aragora.server.auth": mock_mod}):
            h = _make_handler()
            body = _body(self._call(h))
            rl = body["checks"]["rate_limiters"]
            assert rl["healthy"] is True
            assert rl["active_ips"] == 10

    # -- Security services in production --

    @patch.dict("os.environ", {"ARAGORA_ENV": "production"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    def test_security_not_configured_in_production_degrades(self, _ai, _redis, _fs):
        """Missing encryption in production -> degraded."""
        with patch(_P_SEC, return_value={"healthy": True, "encryption_configured": False}):
            h = _make_handler()
            result = self._call(h)
            assert _status(result) == 503

    # -- Version --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_version_included(self, _sec, _ai, _redis, _fs):
        """Version is included in the response."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["version"] is not None

    # -- Timestamp --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_timestamp_is_utc(self, _sec, _ai, _redis, _fs):
        """Timestamp ends with Z (UTC)."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["timestamp"].endswith("Z")

    # -- Uptime --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_uptime_is_positive(self, _sec, _ai, _redis, _fs):
        """Uptime seconds should be non-negative."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["uptime_seconds"] >= 0


# ============================================================================
# TestWebsocketHealth - websocket_health() function
# ============================================================================


class TestWebsocketHealth:
    """Tests for websocket_health() - basic WebSocket health endpoint."""

    def _call(self, handler):
        return handler._websocket_health()

    def test_no_ws_manager(self):
        """No ws_manager -> unavailable with 200."""
        h = _make_handler()
        result = self._call(h)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "unavailable"
        assert body["clients"] == 0

    def test_no_ws_manager_message(self):
        """No ws_manager -> message indicates not configured."""
        h = _make_handler()
        body = _body(self._call(h))
        assert "message" in body

    def test_healthy_ws(self):
        """WS manager present with clients."""
        ws = MagicMock()
        ws.clients = ["c1", "c2", "c3"]
        h = _make_handler({"ws_manager": ws})
        result = self._call(h)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["clients"] == 3

    def test_ws_zero_clients(self):
        """WS manager with zero clients."""
        ws = MagicMock()
        ws.clients = []
        h = _make_handler({"ws_manager": ws})
        result = self._call(h)
        body = _body(result)
        assert _status(result) == 200
        assert body["clients"] == 0

    def test_ws_error_returns_503(self):
        """WS manager raises RuntimeError -> 503 error."""
        ws = MagicMock()
        type(ws).clients = PropertyMock(side_effect=RuntimeError("ws broken"))
        h = _make_handler({"ws_manager": ws})
        result = self._call(h)
        assert _status(result) == 503
        body = _body(result)
        assert body["status"] == "error"

    def test_ws_attribute_error_fallback(self):
        """WS manager with no 'clients' attr uses getattr default -> healthy with 0."""
        # websocket_health uses getattr(ws, "clients", []) so AttributeError
        # is swallowed by getattr's default. This tests the fallback path.
        ws = MagicMock(spec=[])  # empty spec: no .clients attribute
        h = _make_handler({"ws_manager": ws})
        result = self._call(h)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["clients"] == 0

    def test_ws_type_error(self):
        """WS manager raises TypeError -> 503."""
        ws = MagicMock()
        type(ws).clients = PropertyMock(side_effect=TypeError("wrong type"))
        h = _make_handler({"ws_manager": ws})
        result = self._call(h)
        assert _status(result) == 503


# ============================================================================
# TestDetailedHealthCheck - detailed_health_check() function
# ============================================================================


class TestDetailedHealthCheck:
    """Tests for detailed_health_check() with observer metrics."""

    def _call(self, handler):
        return handler._detailed_health_check()

    def test_basic_response_structure(self):
        """Response has required top-level keys."""
        h = _make_handler()
        body = _body(self._call(h))
        for key in ("status", "components", "version", "warnings"):
            assert key in body, f"Missing key: {key}"

    def test_components_reflect_context(self):
        """Components dict reflects what is available in context."""
        mock_storage = MagicMock()
        mock_elo = MagicMock()
        h = _make_handler({"storage": mock_storage, "elo_system": mock_elo})
        body = _body(self._call(h))
        assert body["components"]["storage"] is True
        assert body["components"]["elo_system"] is True

    def test_components_missing_storage(self):
        """Missing storage -> components.storage is False."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["components"]["storage"] is False

    def test_components_nomic_dir_exists(self, tmp_nomic_dir):
        """nomic_dir exists -> components.nomic_dir is True."""
        h = _make_handler({"nomic_dir": tmp_nomic_dir})
        body = _body(self._call(h))
        assert body["components"]["nomic_dir"] is True

    def test_components_nomic_dir_missing(self):
        """No nomic_dir -> components.nomic_dir is False."""
        h = _make_handler()
        body = _body(self._call(h))
        assert body["components"]["nomic_dir"] is False

    # -- Observer metrics --

    def test_observer_unavailable(self):
        """SimpleObserver not importable -> observer status unavailable."""
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": None}):
            h = _make_handler()
            body = _body(self._call(h))
            if "observer" in body:
                assert body["observer"]["status"] in ("unavailable", "error")

    def test_observer_high_failure_rate_degrades(self):
        """Failure rate > 50% -> status = degraded."""
        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {"failure_rate": 0.6, "total_calls": 100}
        mock_module = MagicMock()
        mock_module.SimpleObserver.return_value = mock_observer
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            assert body["status"] == "degraded"
            assert any("High failure rate" in w for w in body.get("warnings", []))

    def test_observer_elevated_failure_rate_warns(self):
        """Failure rate 30-50% -> warning added."""
        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {"failure_rate": 0.35, "total_calls": 100}
        mock_module = MagicMock()
        mock_module.SimpleObserver.return_value = mock_observer
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            assert any("Elevated failure rate" in w for w in body.get("warnings", []))

    def test_observer_low_failure_rate_healthy(self):
        """Failure rate < 30% -> no warnings from observer."""
        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {"failure_rate": 0.1, "total_calls": 100}
        mock_module = MagicMock()
        mock_module.SimpleObserver.return_value = mock_observer
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            assert body["status"] == "healthy"

    def test_observer_error_in_report(self):
        """Observer report with 'error' key -> not added to health."""
        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {"error": "log file missing"}
        mock_module = MagicMock()
        mock_module.SimpleObserver.return_value = mock_observer
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            if "observer" in body:
                assert "failure_rate" not in body.get("observer", {})

    def test_observer_os_error(self):
        """Observer raises OSError -> graceful degradation."""
        mock_module = MagicMock()
        mock_module.SimpleObserver.side_effect = OSError("file not found")
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            if "observer" in body:
                assert body["observer"]["status"] == "error"

    # -- SQLite production warning --

    @patch.dict("os.environ", {"ARAGORA_ENV": "production", "DATABASE_URL": ""})
    def test_sqlite_in_production_warning(self):
        """SQLite in production -> warning added."""
        h = _make_handler()
        body = _body(self._call(h))
        assert any("SQLite" in w for w in body.get("warnings", []))
        assert body["database"]["production_ready"] is False

    @patch.dict(
        "os.environ",
        {"ARAGORA_ENV": "production", "DATABASE_URL": "postgresql://localhost/aragora"},
    )
    def test_postgres_in_production_no_warning(self):
        """PostgreSQL in production -> no SQLite warning."""
        h = _make_handler()
        body = _body(self._call(h))
        sqlite_warnings = [w for w in body.get("warnings", []) if "SQLite" in w]
        assert len(sqlite_warnings) == 0
        assert body["database"]["type"] == "postgresql"
        assert body["database"]["production_ready"] is True

    @patch.dict("os.environ", {"ARAGORA_ENV": "development", "DATABASE_URL": ""})
    def test_sqlite_in_development_no_warning(self):
        """SQLite in development -> no warning."""
        h = _make_handler()
        body = _body(self._call(h))
        sqlite_warnings = [w for w in body.get("warnings", []) if "SQLite" in w]
        assert len(sqlite_warnings) == 0

    # -- Memory stats --

    def test_memory_stats_with_psutil(self):
        """Memory stats included when psutil available."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process.memory_percent.return_value = 5.0
        mock_psutil = MagicMock()
        mock_psutil.Process.return_value = mock_process
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            h = _make_handler()
            body = _body(self._call(h))
            assert "memory" in body
            assert body["memory"]["rss_mb"] == 100.0

    def test_memory_stats_without_psutil(self):
        """No psutil -> no memory stats (graceful)."""
        with patch.dict("sys.modules", {"psutil": None}):
            h = _make_handler()
            body = _body(self._call(h))
            assert body["status"] in ("healthy", "degraded")

    # -- HTTP connector --

    def test_http_connector_healthy(self):
        """HTTP connector open -> healthy."""
        mock_connector = MagicMock()
        mock_connector.closed = False
        mock_module = MagicMock()
        mock_module.get_shared_connector.return_value = mock_connector
        with patch.dict("sys.modules", {"aragora.agents.api_agents.common": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            assert body["http_connector"]["status"] == "healthy"

    def test_http_connector_closed_degrades(self):
        """HTTP connector closed -> degraded."""
        mock_connector = MagicMock()
        mock_connector.closed = True
        mock_module = MagicMock()
        mock_module.get_shared_connector.return_value = mock_connector
        with patch.dict("sys.modules", {"aragora.agents.api_agents.common": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            assert body["http_connector"]["status"] == "closed"
            assert body["status"] == "degraded"

    def test_http_connector_import_error(self):
        """HTTP connector module missing -> unavailable."""
        with patch.dict("sys.modules", {"aragora.agents.api_agents.common": None}):
            h = _make_handler()
            body = _body(self._call(h))
            if "http_connector" in body:
                assert body["http_connector"]["status"] in ("unavailable", "error")

    # -- Export cache --

    def test_export_cache_healthy(self):
        """Export cache reports entries."""
        mock_module = MagicMock()
        mock_module.get_export_cache_stats.return_value = {"total_entries": 42}
        with patch.dict("sys.modules", {"aragora.visualization.exporter": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            assert body["export_cache"]["status"] == "healthy"
            assert body["export_cache"]["entries"] == 42

    def test_export_cache_import_error(self):
        """Export cache module missing -> unavailable."""
        with patch.dict("sys.modules", {"aragora.visualization.exporter": None}):
            h = _make_handler()
            body = _body(self._call(h))
            if "export_cache" in body:
                assert body["export_cache"]["status"] in ("unavailable", "error")

    # -- Handler cache --

    def test_handler_cache_healthy(self):
        """Handler cache returns stats."""
        mock_module = MagicMock()
        mock_module.get_cache_stats.return_value = {"hits": 10, "misses": 2}
        with patch.dict("sys.modules", {"aragora.server.handlers.admin.cache": mock_module}):
            h = _make_handler()
            body = _body(self._call(h))
            assert body["handler_cache"]["status"] == "healthy"

    # -- Return format --

    def test_returns_200(self):
        """Detailed health check returns 200 by default."""
        h = _make_handler()
        result = self._call(h)
        assert _status(result) == 200

    def test_version_is_string(self):
        """Version field is a string."""
        h = _make_handler()
        body = _body(self._call(h))
        assert isinstance(body["version"], str)


# ============================================================================
# TestDeepHealthCheck - deep_health_check() function
# ============================================================================


class TestDeepHealthCheck:
    """Tests for deep_health_check() - full dependency verification."""

    def _call(self, handler):
        return handler._deep_health_check()

    def _make_deep_handler(self, ctx=None):
        """Create handler with given context."""
        return _make_handler(ctx or {})

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_healthy_response(self, _slack, _stripe, _ai, _redis, _fs):
        """All checks pass -> status=healthy, 200."""
        h = self._make_deep_handler()
        result = self._call(h)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] in ("healthy", "healthy_with_warnings")

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_response_has_required_fields(self, _slack, _stripe, _ai, _redis, _fs):
        """Response contains standard fields."""
        h = self._make_deep_handler()
        body = _body(self._call(h))
        for key in ("status", "version", "checks", "response_time_ms", "timestamp"):
            assert key in body, f"Missing key: {key}"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_timestamp_is_utc(self, _slack, _stripe, _ai, _redis, _fs):
        """Timestamp ends with Z."""
        h = self._make_deep_handler()
        body = _body(self._call(h))
        assert body["timestamp"].endswith("Z")

    # -- Storage deep check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_storage_connected(self, _slack, _stripe, _ai, _redis, _fs):
        """Storage accessible -> connected."""
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = []
        h = self._make_deep_handler({"storage": mock_storage})
        body = _body(self._call(h))
        assert body["checks"]["storage"]["healthy"] is True
        assert body["checks"]["storage"]["status"] == "connected"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_storage_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """No storage -> not_configured (still healthy)."""
        h = self._make_deep_handler()
        body = _body(self._call(h))
        assert body["checks"]["storage"]["healthy"] is True
        assert body["checks"]["storage"]["status"] == "not_configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_storage_error_degrades(self, _slack, _stripe, _ai, _redis, _fs):
        """Storage raises -> unhealthy, status=degraded."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = RuntimeError("db error")
        h = self._make_deep_handler({"storage": mock_storage})
        body = _body(self._call(h))
        assert body["checks"]["storage"]["healthy"] is False
        assert body["status"] == "degraded"

    # -- ELO deep check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_elo_connected(self, _slack, _stripe, _ai, _redis, _fs):
        """ELO system accessible -> connected."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        h = self._make_deep_handler({"elo_system": mock_elo})
        body = _body(self._call(h))
        assert body["checks"]["elo_system"]["healthy"] is True

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_elo_error_degrades(self, _slack, _stripe, _ai, _redis, _fs):
        """ELO raises -> unhealthy, degrades."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = OSError("elo fs error")
        h = self._make_deep_handler({"elo_system": mock_elo})
        body = _body(self._call(h))
        assert body["checks"]["elo_system"]["healthy"] is False
        assert body["status"] == "degraded"

    # -- Supabase check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_supabase_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """Supabase not configured -> healthy with not_configured status."""
        mock_client = MagicMock()
        mock_client.is_configured = False
        mock_client.client = None
        mock_module = MagicMock()
        mock_module.SupabaseClient.return_value = mock_client
        with patch.dict("sys.modules", {"aragora.persistence.supabase_client": mock_module}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["supabase"]["healthy"] is True
            assert body["checks"]["supabase"]["status"] == "not_configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_supabase_import_error(self, _slack, _stripe, _ai, _redis, _fs):
        """Supabase module missing -> module_not_available."""
        with patch.dict("sys.modules", {"aragora.persistence.supabase_client": None}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["supabase"]["status"] == "module_not_available"

    # -- User store check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_user_store_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """No user store -> not_configured."""
        h = self._make_deep_handler()
        body = _body(self._call(h))
        assert body["checks"]["user_store"]["healthy"] is True
        assert body["checks"]["user_store"]["status"] == "not_configured"

    # -- Billing check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_billing_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """Billing module present but not configured."""
        mock_client = MagicMock()
        mock_client._is_configured.return_value = False
        mock_module = MagicMock()
        mock_module.StripeClient.return_value = mock_client
        with patch.dict("sys.modules", {"aragora.billing.stripe_client": mock_module}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["billing"]["status"] == "not_configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_billing_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """Billing configured -> status configured."""
        mock_client = MagicMock()
        mock_client._is_configured.return_value = True
        mock_module = MagicMock()
        mock_module.StripeClient.return_value = mock_client
        with patch.dict("sys.modules", {"aragora.billing.stripe_client": mock_module}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["billing"]["status"] == "configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_billing_import_error(self, _slack, _stripe, _ai, _redis, _fs):
        """Billing module missing -> module_not_available."""
        with patch.dict("sys.modules", {"aragora.billing.stripe_client": None}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["billing"]["status"] == "module_not_available"

    # -- Filesystem deep check --

    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_filesystem_failure_degrades(self, _slack, _stripe, _ai, _redis):
        """Filesystem failure in deep check -> degraded."""
        with patch(_P_FS, return_value={"healthy": False, "error": "Permission denied"}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["status"] == "degraded"

    # -- System resources (psutil) --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_high_memory_usage_degrades(self, _slack, _stripe, _ai, _redis, _fs):
        """Memory >=90% -> degraded."""
        mock_psutil = MagicMock()
        mock_memory = MagicMock()
        mock_memory.percent = 95.0
        mock_memory.available = 1 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_disk.free = 100 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["memory"]["healthy"] is False
            assert body["status"] == "degraded"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_elevated_memory_warning(self, _slack, _stripe, _ai, _redis, _fs):
        """Memory 80-89% -> warning."""
        mock_psutil = MagicMock()
        mock_memory = MagicMock()
        mock_memory.percent = 85.0
        mock_memory.available = 2 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_disk.free = 100 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            warnings = body.get("warnings") or []
            assert any("Elevated memory" in w for w in warnings)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_high_disk_usage_degrades(self, _slack, _stripe, _ai, _redis, _fs):
        """Disk >=90% -> degraded."""
        mock_psutil = MagicMock()
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 95.0
        mock_disk.free = 5 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["disk"]["healthy"] is False
            assert body["status"] == "degraded"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_disk_space_warning(self, _slack, _stripe, _ai, _redis, _fs):
        """Disk 80-89% -> warning."""
        mock_psutil = MagicMock()
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 85.0
        mock_disk.free = 15 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            warnings = body.get("warnings") or []
            assert any("Disk space warning" in w for w in warnings)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_high_cpu_warning(self, _slack, _stripe, _ai, _redis, _fs):
        """CPU >=90% -> warning."""
        mock_psutil = MagicMock()
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_disk.free = 100 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            warnings = body.get("warnings") or []
            assert any("High CPU" in w for w in warnings)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_psutil_not_available(self, _slack, _stripe, _ai, _redis, _fs):
        """No psutil -> system_resources with psutil_not_available."""
        with patch.dict("sys.modules", {"psutil": None}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            if "system_resources" in body["checks"]:
                assert body["checks"]["system_resources"]["status"] == "psutil_not_available"
            assert body["status"] in ("healthy", "healthy_with_warnings")

    # -- Stripe check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_stripe_configured_and_failing(self, _slack, _ai, _redis, _fs):
        """Stripe configured but failing -> degraded + warning."""
        with patch(
            _P_STRIPE, return_value={"healthy": False, "configured": True, "error": "Auth failed"}
        ):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["status"] == "degraded"
            warnings = body.get("warnings") or []
            assert any("Stripe" in w for w in warnings)

    # -- Slack check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    def test_slack_configured_and_failing(self, _stripe, _ai, _redis, _fs):
        """Slack configured but failing -> degraded + warning."""
        with patch(
            _P_SLACK,
            return_value={"healthy": False, "configured": True, "error": "Connection failed"},
        ):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["status"] == "degraded"
            warnings = body.get("warnings") or []
            assert any("Slack" in w for w in warnings)

    # -- Healthy with warnings status --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_healthy_with_warnings_status(self, _slack, _stripe, _redis, _fs):
        """Warnings but no failures -> healthy_with_warnings."""
        with patch(
            _P_AI, return_value={"healthy": True, "any_available": False, "available_count": 0}
        ):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["status"] == "healthy_with_warnings"
            warnings = body.get("warnings") or []
            assert any("No AI providers" in w for w in warnings)

    # -- Email services --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_email_services_not_available(self, _slack, _stripe, _ai, _redis, _fs):
        """Email services module missing -> not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.services.followup_tracker": None,
                "aragora.services.snooze_recommender": None,
            },
        ):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["email_services"]["status"] == "not_available"

    # -- Dependency analyzer --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_dependency_analyzer_not_available(self, _slack, _stripe, _ai, _redis, _fs):
        """Dependency analyzer module missing -> not_available."""
        with patch.dict("sys.modules", {"aragora.audit.dependency_analyzer": None}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["checks"]["dependency_analyzer"]["status"] == "not_available"

    # -- Warnings list is None when empty --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_warnings_none_when_empty(self, _slack, _stripe, _ai, _redis, _fs):
        """No warnings -> warnings field is None."""
        # Mock psutil so real system stats don't add warnings
        mock_psutil = MagicMock()
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_disk.free = 100 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body.get("warnings") is None

    # -- Redis deep check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_redis_configured_failing_degrades(self, _slack, _stripe, _ai, _fs):
        """Redis configured but unhealthy -> degrades."""
        with patch(
            _P_REDIS,
            return_value={"healthy": False, "configured": True, "error": "Connection refused"},
        ):
            h = self._make_deep_handler()
            body = _body(self._call(h))
            assert body["status"] == "degraded"

    # -- Version --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_version_included(self, _slack, _stripe, _ai, _redis, _fs):
        """Version field is present."""
        h = self._make_deep_handler()
        body = _body(self._call(h))
        assert "version" in body
        assert isinstance(body["version"], str)

    # -- Response time --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_response_time_is_numeric(self, _slack, _stripe, _ai, _redis, _fs):
        """response_time_ms is a number."""
        h = self._make_deep_handler()
        body = _body(self._call(h))
        assert isinstance(body["response_time_ms"], (int, float))
        assert body["response_time_ms"] >= 0
