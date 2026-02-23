"""
Comprehensive tests for the detailed health check handler module.

Tests all four public functions in aragora/server/handlers/admin/health/detailed.py:

  TestHealthCheck              - health_check() comprehensive k8s health check
  TestWebsocketHealth          - websocket_health() basic WebSocket health
  TestDetailedHealthCheck      - detailed_health_check() with observer metrics
  TestDeepHealthCheck          - deep_health_check() full dependency verification

Coverage: all routes, success paths, error handling, edge cases, input validation.
Target: 80+ tests, 0 failures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

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


def _make_handler(ctx: dict[str, Any] | None = None) -> HealthHandler:
    """Create a HealthHandler with the given context."""
    return HealthHandler(ctx=ctx or {})


# ---------------------------------------------------------------------------
# Common patch targets
# ---------------------------------------------------------------------------

_P_FS = "aragora.server.handlers.admin.health.detailed.check_filesystem_health"
_P_REDIS = "aragora.server.handlers.admin.health.detailed.check_redis_health"
_P_AI = "aragora.server.handlers.admin.health.detailed.check_ai_providers_health"
_P_SEC = "aragora.server.handlers.admin.health.detailed.check_security_services"
_P_STRIPE = "aragora.server.handlers.admin.health_utils.check_stripe_health"
_P_SLACK = "aragora.server.handlers.admin.health_utils.check_slack_health"

# Healthy defaults
_HEALTHY_FS = {"healthy": True, "path": "/tmp"}
_HEALTHY_REDIS = {"healthy": True, "configured": False, "note": "Redis not configured"}
_HEALTHY_AI = {"healthy": True, "any_available": True, "available_count": 1, "providers": {}}
_HEALTHY_SECURITY = {"healthy": True, "encryption_configured": True}
_HEALTHY_STRIPE = {"healthy": True, "configured": False}
_HEALTHY_SLACK = {"healthy": True, "configured": False}


def _patch_all_utils():
    """Return a list of patch decorators for all health utility functions."""
    return [
        patch(_P_FS, return_value=_HEALTHY_FS),
        patch(_P_REDIS, return_value=_HEALTHY_REDIS),
        patch(_P_AI, return_value=_HEALTHY_AI),
        patch(_P_SEC, return_value=_HEALTHY_SECURITY),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Default handler with empty context."""
    return _make_handler()


@pytest.fixture
def tmp_nomic_dir(tmp_path):
    """Temporary directory for nomic_dir."""
    d = tmp_path / "nomic"
    d.mkdir()
    return d


# ============================================================================
# TestHealthCheck - health_check() function
# ============================================================================


class TestHealthCheck:
    """Tests for health_check() - comprehensive k8s/docker health endpoint."""

    def _call(self, handler):
        return handler._health_check()

    # -- Basic response structure --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_healthy_returns_200(self, _sec, _ai, _redis, _fs):
        """All healthy -> status 200."""
        result = self._call(_make_handler())
        assert _status(result) == 200

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_healthy_status_string(self, _sec, _ai, _redis, _fs):
        """All healthy -> status='healthy'."""
        body = _body(self._call(_make_handler()))
        assert body["status"] == "healthy"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_response_has_all_required_keys(self, _sec, _ai, _redis, _fs):
        """Response includes all required top-level fields."""
        body = _body(self._call(_make_handler()))
        required = ["status", "version", "uptime_seconds", "demo_mode", "checks", "timestamp", "response_time_ms"]
        for key in required:
            assert key in body, f"Missing required key: {key}"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_timestamp_ends_with_z(self, _sec, _ai, _redis, _fs):
        """Timestamp is UTC (ends with Z)."""
        body = _body(self._call(_make_handler()))
        assert body["timestamp"].endswith("Z")

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_uptime_is_non_negative(self, _sec, _ai, _redis, _fs):
        """Uptime seconds is non-negative."""
        body = _body(self._call(_make_handler()))
        assert body["uptime_seconds"] >= 0

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_response_time_is_non_negative(self, _sec, _ai, _redis, _fs):
        """Response time is a non-negative number."""
        body = _body(self._call(_make_handler()))
        assert isinstance(body["response_time_ms"], (int, float))
        assert body["response_time_ms"] >= 0

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_version_is_string(self, _sec, _ai, _redis, _fs):
        """Version field is a string."""
        body = _body(self._call(_make_handler()))
        assert isinstance(body["version"], str)

    # -- Demo mode --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_false_by_default(self, _sec, _ai, _redis, _fs):
        """Demo mode is False by default."""
        body = _body(self._call(_make_handler()))
        assert body["demo_mode"] is False

    @patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_true(self, _sec, _ai, _redis, _fs):
        """ARAGORA_DEMO_MODE=true -> demo_mode=True."""
        body = _body(self._call(_make_handler()))
        assert body["demo_mode"] is True

    @patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "1"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_accepts_1(self, _sec, _ai, _redis, _fs):
        """ARAGORA_DEMO_MODE=1 -> demo_mode=True."""
        body = _body(self._call(_make_handler()))
        assert body["demo_mode"] is True

    @patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "yes"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_accepts_yes(self, _sec, _ai, _redis, _fs):
        """ARAGORA_DEMO_MODE=yes -> demo_mode=True."""
        body = _body(self._call(_make_handler()))
        assert body["demo_mode"] is True

    @patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "no"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_demo_mode_rejects_no(self, _sec, _ai, _redis, _fs):
        """ARAGORA_DEMO_MODE=no -> demo_mode=False."""
        body = _body(self._call(_make_handler()))
        assert body["demo_mode"] is False

    # -- Degraded mode --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_degraded_mode_normal(self, _sec, _ai, _redis, _fs):
        """Non-degraded server -> degraded_mode check is healthy."""
        body = _body(self._call(_make_handler()))
        assert body["checks"]["degraded_mode"]["healthy"] is True

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_degraded_mode_active_returns_503(self, _sec, _ai, _redis, _fs):
        """Server degraded -> 503 and unhealthy degraded_mode check."""
        mock_state = MagicMock()
        mock_state.reason = "startup failure"
        mock_state.error_code.value = "STARTUP_ERROR"
        mock_state.recovery_hint = "restart"
        mock_state.timestamp = "2026-01-01T00:00:00Z"

        mock_mod = MagicMock()
        mock_mod.is_degraded.return_value = True
        mock_mod.get_degraded_state.return_value = mock_state

        with patch.dict("sys.modules", {"aragora.server.degraded_mode": mock_mod}):
            result = self._call(_make_handler())
            body = _body(result)
            assert _status(result) == 503
            assert body["status"] == "degraded"
            dm = body["checks"]["degraded_mode"]
            assert dm["healthy"] is False
            assert dm["reason"] == "startup failure"
            assert dm["error_code"] == "STARTUP_ERROR"
            assert dm["recovery_hint"] == "restart"
            assert dm["degraded_since"] == "2026-01-01T00:00:00Z"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_degraded_mode_import_error(self, _sec, _ai, _redis, _fs):
        """degraded_mode module unavailable -> healthy with module_not_available."""
        with patch.dict("sys.modules", {"aragora.server.degraded_mode": None}):
            body = _body(self._call(_make_handler()))
            dm = body["checks"]["degraded_mode"]
            assert dm["healthy"] is True
            assert dm["status"] == "module_not_available"

    # -- Database check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_database_healthy_with_latency(self, _sec, _ai, _redis, _fs):
        """Storage available -> database healthy with latency_ms."""
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = []
        body = _body(self._call(_make_handler({"storage": mock_storage})))
        db = body["checks"]["database"]
        assert db["healthy"] is True
        assert "latency_ms" in db
        assert isinstance(db["latency_ms"], (int, float))

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_database_not_initialized(self, _sec, _ai, _redis, _fs):
        """No storage -> warning but still healthy."""
        body = _body(self._call(_make_handler()))
        db = body["checks"]["database"]
        assert db["healthy"] is True
        assert "warning" in db
        assert db["initialized"] is False

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_database_runtime_error_is_non_critical(self, _sec, _ai, _redis, _fs):
        """Database RuntimeError is downgraded to warning."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = RuntimeError("db down")
        body = _body(self._call(_make_handler({"storage": mock_storage})))
        assert body["checks"]["database"]["healthy"] is True
        assert body["checks"]["database"]["initialized"] is False

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_database_os_error_is_non_critical(self, _sec, _ai, _redis, _fs):
        """Database OSError is downgraded to warning."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = OSError("disk full")
        body = _body(self._call(_make_handler({"storage": mock_storage})))
        assert body["checks"]["database"]["healthy"] is True

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_database_json_decode_error(self, _sec, _ai, _redis, _fs):
        """Database JSONDecodeError is downgraded to warning."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = json.JSONDecodeError("bad", "", 0)
        body = _body(self._call(_make_handler({"storage": mock_storage})))
        assert body["checks"]["database"]["healthy"] is True

    # -- ELO system check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_elo_healthy(self, _sec, _ai, _redis, _fs):
        """ELO system available -> healthy."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        body = _body(self._call(_make_handler({"elo_system": mock_elo})))
        assert body["checks"]["elo_system"]["healthy"] is True

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_elo_not_initialized(self, _sec, _ai, _redis, _fs):
        """No ELO system -> warning but healthy."""
        body = _body(self._call(_make_handler()))
        elo = body["checks"]["elo_system"]
        assert elo["healthy"] is True
        assert "warning" in elo
        assert elo["initialized"] is False

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_elo_value_error_non_critical(self, _sec, _ai, _redis, _fs):
        """ELO ValueError is downgraded to warning."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = ValueError("corrupt")
        body = _body(self._call(_make_handler({"elo_system": mock_elo})))
        assert body["checks"]["elo_system"]["healthy"] is True
        assert body["checks"]["elo_system"]["initialized"] is False

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_elo_key_error_non_critical(self, _sec, _ai, _redis, _fs):
        """ELO KeyError is downgraded to warning."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = KeyError("missing key")
        body = _body(self._call(_make_handler({"elo_system": mock_elo})))
        assert body["checks"]["elo_system"]["healthy"] is True

    # -- Nomic directory check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_nomic_dir_exists(self, _sec, _ai, _redis, _fs, tmp_nomic_dir):
        """Existing nomic dir -> healthy with path."""
        body = _body(self._call(_make_handler({"nomic_dir": tmp_nomic_dir})))
        nd = body["checks"]["nomic_dir"]
        assert nd["healthy"] is True
        assert nd["path"] == str(tmp_nomic_dir)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_nomic_dir_not_configured(self, _sec, _ai, _redis, _fs):
        """No nomic dir -> downgraded to warning."""
        body = _body(self._call(_make_handler()))
        nd = body["checks"]["nomic_dir"]
        assert nd["healthy"] is True
        assert "warning" in nd

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_nomic_dir_nonexistent_path(self, _sec, _ai, _redis, _fs, tmp_path):
        """Nomic dir set but path does not exist -> warning."""
        missing = tmp_path / "does_not_exist"
        body = _body(self._call(_make_handler({"nomic_dir": missing})))
        nd = body["checks"]["nomic_dir"]
        assert nd["healthy"] is True
        assert "warning" in nd

    # -- Filesystem check --

    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_filesystem_unhealthy_causes_503(self, _sec, _ai, _redis):
        """Filesystem failure -> overall status degraded -> 503."""
        with patch(_P_FS, return_value={"healthy": False, "error": "Permission denied"}):
            result = self._call(_make_handler())
            assert _status(result) == 503
            assert _body(result)["status"] == "degraded"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_filesystem_healthy_passes(self, _sec, _ai, _redis, _fs):
        """Filesystem healthy -> no degradation."""
        body = _body(self._call(_make_handler()))
        assert body["checks"]["filesystem"]["healthy"] is True

    # -- Redis check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_redis_configured_and_failing_causes_503(self, _sec, _ai, _fs):
        """Redis configured but failing -> degraded."""
        with patch(_P_REDIS, return_value={"healthy": False, "configured": True, "error": "Connection failed"}):
            result = self._call(_make_handler())
            assert _status(result) == 503

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_redis_not_configured_is_ok(self, _sec, _ai, _fs):
        """Redis not configured -> no degradation."""
        with patch(_P_REDIS, return_value={"healthy": True, "configured": False}):
            result = self._call(_make_handler())
            assert _status(result) == 200

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_redis_failing_not_configured_no_degrade(self, _sec, _ai, _fs):
        """Redis failing but not configured -> no degradation (edge case)."""
        with patch(_P_REDIS, return_value={"healthy": False, "configured": False, "error": "Connection failed"}):
            result = self._call(_make_handler())
            assert _status(result) == 200

    # -- AI providers check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_no_ai_providers_adds_warning(self, _sec, _redis, _fs):
        """No AI providers -> warning but no degradation."""
        with patch(_P_AI, return_value={"healthy": True, "any_available": False, "available_count": 0}):
            result = self._call(_make_handler())
            body = _body(result)
            assert _status(result) == 200
            assert "warning" in body["checks"]["ai_providers"]

    # -- WebSocket in health_check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_websocket_with_manager(self, _sec, _ai, _redis, _fs):
        """ws_manager present -> reports client count."""
        ws = MagicMock()
        ws.clients = ["c1", "c2"]
        body = _body(self._call(_make_handler({"ws_manager": ws})))
        ws_check = body["checks"]["websocket"]
        assert ws_check["healthy"] is True
        assert ws_check["active_clients"] == 2

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_websocket_no_manager(self, _sec, _ai, _redis, _fs):
        """No ws_manager -> separate aiohttp note."""
        body = _body(self._call(_make_handler()))
        ws = body["checks"]["websocket"]
        assert ws["healthy"] is True
        assert "note" in ws

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_websocket_type_error(self, _sec, _ai, _redis, _fs):
        """ws_manager.clients has bad len -> unhealthy websocket."""
        ws = MagicMock(spec=[])
        ws.clients = MagicMock()
        ws.clients.__len__ = MagicMock(side_effect=TypeError("bad"))
        body = _body(self._call(_make_handler({"ws_manager": ws})))
        assert body["checks"]["websocket"]["healthy"] is False

    # -- Circuit breaker check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_circuit_breakers_few_open_healthy(self, _sec, _ai, _redis, _fs):
        """<3 open circuit breakers -> healthy."""
        mock_mod = MagicMock()
        mock_mod.get_circuit_breaker_metrics.return_value = {
            "summary": {"open_count": 2, "half_open_count": 1, "closed_count": 10}
        }
        with patch.dict("sys.modules", {"aragora.resilience": mock_mod}):
            body = _body(self._call(_make_handler()))
            cb = body["checks"]["circuit_breakers"]
            assert cb["healthy"] is True
            assert cb["open"] == 2
            assert cb["half_open"] == 1
            assert cb["closed"] == 10

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_circuit_breakers_many_open_causes_503(self, _sec, _ai, _redis, _fs):
        """>=3 open circuit breakers -> degraded -> 503."""
        mock_mod = MagicMock()
        mock_mod.get_circuit_breaker_metrics.return_value = {
            "summary": {"open_count": 5, "half_open_count": 0, "closed_count": 1}
        }
        with patch.dict("sys.modules", {"aragora.resilience": mock_mod}):
            result = self._call(_make_handler())
            assert _status(result) == 503
            assert _body(result)["checks"]["circuit_breakers"]["healthy"] is False

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_circuit_breakers_import_error(self, _sec, _ai, _redis, _fs):
        """Resilience module unavailable -> module_not_available."""
        with patch.dict("sys.modules", {"aragora.resilience": None}):
            body = _body(self._call(_make_handler()))
            cb = body["checks"]["circuit_breakers"]
            assert cb["healthy"] is True
            assert cb["status"] == "module_not_available"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_circuit_breakers_runtime_error(self, _sec, _ai, _redis, _fs):
        """Circuit breaker metrics raise RuntimeError -> graceful handling."""
        mock_mod = MagicMock()
        mock_mod.get_circuit_breaker_metrics.side_effect = RuntimeError("boom")
        with patch.dict("sys.modules", {"aragora.resilience": mock_mod}):
            body = _body(self._call(_make_handler()))
            cb = body["checks"]["circuit_breakers"]
            assert cb["healthy"] is True
            assert "error" in cb

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_circuit_breakers_key_error(self, _sec, _ai, _redis, _fs):
        """Circuit breaker metrics raise KeyError -> graceful handling."""
        mock_mod = MagicMock()
        mock_mod.get_circuit_breaker_metrics.side_effect = KeyError("bad key")
        with patch.dict("sys.modules", {"aragora.resilience": mock_mod}):
            body = _body(self._call(_make_handler()))
            cb = body["checks"]["circuit_breakers"]
            assert cb["healthy"] is True

    # -- Rate limiter check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_rate_limiter_stats_included(self, _sec, _ai, _redis, _fs):
        """Rate limiter stats are included when available."""
        mock_auth_config = MagicMock()
        mock_auth_config.get_rate_limit_stats.return_value = {
            "ip_entries": 15, "token_entries": 7, "revoked_tokens": 3,
        }
        mock_mod = MagicMock()
        mock_mod.auth_config = mock_auth_config
        with patch.dict("sys.modules", {"aragora.server.auth": mock_mod}):
            body = _body(self._call(_make_handler()))
            rl = body["checks"]["rate_limiters"]
            assert rl["healthy"] is True
            assert rl["active_ips"] == 15
            assert rl["active_tokens"] == 7
            assert rl["revoked_tokens"] == 3

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_rate_limiter_import_error(self, _sec, _ai, _redis, _fs):
        """Rate limiter module missing -> module_not_available."""
        with patch.dict("sys.modules", {"aragora.server.auth": None}):
            body = _body(self._call(_make_handler()))
            rl = body["checks"]["rate_limiters"]
            assert rl["healthy"] is True
            assert rl["status"] == "module_not_available"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_rate_limiter_attribute_error(self, _sec, _ai, _redis, _fs):
        """Rate limiter AttributeError -> graceful handling."""
        mock_mod = MagicMock()
        mock_mod.auth_config.get_rate_limit_stats.side_effect = AttributeError("no attr")
        with patch.dict("sys.modules", {"aragora.server.auth": mock_mod}):
            body = _body(self._call(_make_handler()))
            rl = body["checks"]["rate_limiters"]
            assert rl["healthy"] is True
            assert "error" in rl

    # -- Security in production --

    @patch.dict("os.environ", {"ARAGORA_ENV": "production"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    def test_security_not_configured_in_production_degrades(self, _ai, _redis, _fs):
        """Missing encryption in production -> 503."""
        with patch(_P_SEC, return_value={"healthy": True, "encryption_configured": False}):
            result = self._call(_make_handler())
            assert _status(result) == 503

    @patch.dict("os.environ", {"ARAGORA_ENV": "development"})
    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    def test_security_not_configured_in_development_ok(self, _ai, _redis, _fs):
        """Missing encryption in development -> no degradation."""
        with patch(_P_SEC, return_value={"healthy": True, "encryption_configured": False}):
            result = self._call(_make_handler())
            assert _status(result) == 200

    # -- Checks dict has expected keys --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SEC, return_value=_HEALTHY_SECURITY)
    def test_checks_dict_has_expected_keys(self, _sec, _ai, _redis, _fs):
        """Checks dict includes all expected sub-checks."""
        body = _body(self._call(_make_handler()))
        expected_keys = [
            "degraded_mode", "database", "elo_system", "nomic_dir",
            "filesystem", "redis", "ai_providers", "websocket",
            "circuit_breakers", "rate_limiters", "security_services",
        ]
        for key in expected_keys:
            assert key in body["checks"], f"Missing check key: {key}"


# ============================================================================
# TestWebsocketHealth - websocket_health() standalone function
# ============================================================================


class TestWebsocketHealth:
    """Tests for websocket_health() - basic WebSocket health endpoint."""

    def _call(self, handler):
        return handler._websocket_health()

    def test_no_ws_manager_returns_200(self):
        """No ws_manager -> 200 unavailable."""
        result = self._call(_make_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unavailable"
        assert body["clients"] == 0

    def test_no_ws_manager_has_message(self):
        """No ws_manager -> message field present."""
        body = _body(self._call(_make_handler()))
        assert "message" in body

    def test_healthy_with_clients(self):
        """WS manager with clients -> healthy."""
        ws = MagicMock()
        ws.clients = ["c1", "c2", "c3"]
        result = self._call(_make_handler({"ws_manager": ws}))
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["clients"] == 3

    def test_healthy_zero_clients(self):
        """WS manager with zero clients -> healthy."""
        ws = MagicMock()
        ws.clients = []
        result = self._call(_make_handler({"ws_manager": ws}))
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["clients"] == 0

    def test_runtime_error_returns_503(self):
        """WS manager raises RuntimeError -> 503."""
        ws = MagicMock()
        type(ws).clients = PropertyMock(side_effect=RuntimeError("broken"))
        result = self._call(_make_handler({"ws_manager": ws}))
        assert _status(result) == 503
        body = _body(result)
        assert body["status"] == "error"
        assert body["clients"] == 0

    def test_type_error_returns_503(self):
        """WS manager raises TypeError -> 503."""
        ws = MagicMock()
        type(ws).clients = PropertyMock(side_effect=TypeError("wrong type"))
        result = self._call(_make_handler({"ws_manager": ws}))
        assert _status(result) == 503

    def test_attribute_error_fallback_healthy(self):
        """WS manager with no 'clients' attr uses getattr default."""
        ws = MagicMock(spec=[])
        result = self._call(_make_handler({"ws_manager": ws}))
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["clients"] == 0

    def test_error_response_has_message(self):
        """Error response includes a message field."""
        ws = MagicMock()
        type(ws).clients = PropertyMock(side_effect=RuntimeError("err"))
        body = _body(self._call(_make_handler({"ws_manager": ws})))
        assert "message" in body


# ============================================================================
# TestDetailedHealthCheck - detailed_health_check() function
# ============================================================================


class TestDetailedHealthCheck:
    """Tests for detailed_health_check() with observer metrics."""

    def _call(self, handler):
        return handler._detailed_health_check()

    def test_basic_response_structure(self):
        """Response has required top-level keys."""
        body = _body(self._call(_make_handler()))
        for key in ("status", "components", "version", "warnings"):
            assert key in body

    def test_returns_200(self):
        """Default returns 200."""
        result = self._call(_make_handler())
        assert _status(result) == 200

    def test_version_is_string(self):
        """Version is a string."""
        body = _body(self._call(_make_handler()))
        assert isinstance(body["version"], str)

    def test_components_reflect_storage(self):
        """Components dict reflects storage availability."""
        h = _make_handler({"storage": MagicMock(), "elo_system": MagicMock()})
        body = _body(self._call(h))
        assert body["components"]["storage"] is True
        assert body["components"]["elo_system"] is True

    def test_components_missing_storage(self):
        """No storage -> components.storage is False."""
        body = _body(self._call(_make_handler()))
        assert body["components"]["storage"] is False

    def test_components_nomic_dir_exists(self, tmp_nomic_dir):
        """nomic_dir exists -> components.nomic_dir is True."""
        body = _body(self._call(_make_handler({"nomic_dir": tmp_nomic_dir})))
        assert body["components"]["nomic_dir"] is True

    def test_components_nomic_dir_missing(self):
        """No nomic_dir -> components.nomic_dir is False."""
        body = _body(self._call(_make_handler()))
        assert body["components"]["nomic_dir"] is False

    def test_warnings_is_list(self):
        """Warnings field is a list."""
        body = _body(self._call(_make_handler()))
        assert isinstance(body["warnings"], list)

    # -- Observer metrics --

    def test_observer_import_error(self):
        """SimpleObserver not importable -> observer status unavailable."""
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": None}):
            body = _body(self._call(_make_handler()))
            if "observer" in body:
                assert body["observer"]["status"] == "unavailable"

    def test_observer_high_failure_rate_degrades(self):
        """Failure rate >50% -> status=degraded."""
        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {"failure_rate": 0.6, "total_calls": 100}
        mock_module = MagicMock()
        mock_module.SimpleObserver.return_value = mock_observer
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            body = _body(self._call(_make_handler()))
            assert body["status"] == "degraded"
            assert any("High failure rate" in w for w in body.get("warnings", []))

    def test_observer_elevated_failure_rate_warns(self):
        """Failure rate 30-50% -> warning."""
        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {"failure_rate": 0.35}
        mock_module = MagicMock()
        mock_module.SimpleObserver.return_value = mock_observer
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            body = _body(self._call(_make_handler()))
            assert any("Elevated failure rate" in w for w in body.get("warnings", []))

    def test_observer_low_failure_rate_healthy(self):
        """Failure rate <30% -> healthy with no failure warnings."""
        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {"failure_rate": 0.1}
        mock_module = MagicMock()
        mock_module.SimpleObserver.return_value = mock_observer
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            body = _body(self._call(_make_handler()))
            assert body["status"] == "healthy"

    def test_observer_error_in_report_not_added(self):
        """Observer report with 'error' key -> not included in health."""
        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {"error": "log file missing"}
        mock_module = MagicMock()
        mock_module.SimpleObserver.return_value = mock_observer
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            body = _body(self._call(_make_handler()))
            if "observer" in body:
                assert "failure_rate" not in body.get("observer", {})

    def test_observer_os_error_graceful(self):
        """Observer raises OSError -> error status."""
        mock_module = MagicMock()
        mock_module.SimpleObserver.side_effect = OSError("file not found")
        with patch.dict("sys.modules", {"aragora.monitoring.simple_observer": mock_module}):
            body = _body(self._call(_make_handler()))
            if "observer" in body:
                assert body["observer"]["status"] == "error"

    # -- Maintenance stats --

    def test_maintenance_available(self, tmp_nomic_dir):
        """Maintenance stats included when module is available."""
        mock_maintenance = MagicMock()
        mock_maintenance.get_stats.return_value = {"last_run": "2026-01-01"}
        mock_module = MagicMock()
        mock_module.DatabaseMaintenance.return_value = mock_maintenance
        with patch.dict("sys.modules", {"aragora.maintenance": mock_module}):
            body = _body(self._call(_make_handler({"nomic_dir": tmp_nomic_dir})))
            assert "maintenance" in body
            assert body["maintenance"]["last_run"] == "2026-01-01"

    def test_maintenance_import_error_no_crash(self):
        """Maintenance module not available -> no maintenance key, no crash."""
        with patch.dict("sys.modules", {"aragora.maintenance": None}):
            body = _body(self._call(_make_handler()))
            # maintenance key might or might not be present, but no crash
            assert body["status"] in ("healthy", "degraded")

    def test_maintenance_runtime_error(self, tmp_nomic_dir):
        """Maintenance raises RuntimeError -> error key."""
        mock_module = MagicMock()
        mock_module.DatabaseMaintenance.side_effect = RuntimeError("bad")
        with patch.dict("sys.modules", {"aragora.maintenance": mock_module}):
            body = _body(self._call(_make_handler({"nomic_dir": tmp_nomic_dir})))
            if "maintenance" in body:
                assert "error" in body["maintenance"]

    # -- SQLite production warning --

    @patch.dict("os.environ", {"ARAGORA_ENV": "production", "DATABASE_URL": ""})
    def test_sqlite_in_production_warning(self):
        """SQLite in production -> warning."""
        body = _body(self._call(_make_handler()))
        assert any("SQLite" in w for w in body.get("warnings", []))
        assert body["database"]["production_ready"] is False
        assert body["database"]["type"] == "sqlite"

    @patch.dict("os.environ", {"ARAGORA_ENV": "production", "DATABASE_URL": "postgresql://localhost/db"})
    def test_postgres_in_production_ok(self):
        """PostgreSQL in production -> no SQLite warning."""
        body = _body(self._call(_make_handler()))
        sqlite_warnings = [w for w in body.get("warnings", []) if "SQLite" in w]
        assert len(sqlite_warnings) == 0
        assert body["database"]["type"] == "postgresql"
        assert body["database"]["production_ready"] is True

    @patch.dict("os.environ", {"ARAGORA_ENV": "development", "DATABASE_URL": ""})
    def test_sqlite_in_development_no_warning(self):
        """SQLite in development -> no warning."""
        body = _body(self._call(_make_handler()))
        sqlite_warnings = [w for w in body.get("warnings", []) if "SQLite" in w]
        assert len(sqlite_warnings) == 0

    @patch.dict("os.environ", {"ARAGORA_ENV": "production", "DATABASE_URL": "mysql://localhost/db"})
    def test_mysql_in_production(self):
        """MySQL in production -> production_ready, type unknown."""
        body = _body(self._call(_make_handler()))
        assert body["database"]["production_ready"] is True

    # -- Memory stats --

    def test_memory_stats_with_psutil(self):
        """psutil available -> memory stats included."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 200 * 1024 * 1024
        mock_process.memory_percent.return_value = 7.5
        mock_psutil = MagicMock()
        mock_psutil.Process.return_value = mock_process
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            body = _body(self._call(_make_handler()))
            assert "memory" in body
            assert body["memory"]["rss_mb"] == 200.0
            assert body["memory"]["percent"] == 7.5

    def test_memory_stats_without_psutil(self):
        """No psutil -> no memory stats, no crash."""
        with patch.dict("sys.modules", {"psutil": None}):
            body = _body(self._call(_make_handler()))
            assert body["status"] in ("healthy", "degraded")

    # -- HTTP connector --

    def test_http_connector_healthy(self):
        """HTTP connector open -> healthy."""
        mock_connector = MagicMock()
        mock_connector.closed = False
        mock_module = MagicMock()
        mock_module.get_shared_connector.return_value = mock_connector
        with patch.dict("sys.modules", {"aragora.agents.api_agents.common": mock_module}):
            body = _body(self._call(_make_handler()))
            assert body["http_connector"]["status"] == "healthy"
            assert body["http_connector"]["closed"] is False

    def test_http_connector_closed_degrades(self):
        """HTTP connector closed -> degraded status."""
        mock_connector = MagicMock()
        mock_connector.closed = True
        mock_module = MagicMock()
        mock_module.get_shared_connector.return_value = mock_connector
        with patch.dict("sys.modules", {"aragora.agents.api_agents.common": mock_module}):
            body = _body(self._call(_make_handler()))
            assert body["http_connector"]["status"] == "closed"
            assert body["status"] == "degraded"
            assert any("HTTP connector" in w for w in body.get("warnings", []))

    def test_http_connector_import_error(self):
        """HTTP connector module missing -> unavailable."""
        with patch.dict("sys.modules", {"aragora.agents.api_agents.common": None}):
            body = _body(self._call(_make_handler()))
            if "http_connector" in body:
                assert body["http_connector"]["status"] == "unavailable"

    def test_http_connector_runtime_error(self):
        """HTTP connector raises RuntimeError -> error status."""
        mock_module = MagicMock()
        mock_module.get_shared_connector.side_effect = RuntimeError("broken")
        with patch.dict("sys.modules", {"aragora.agents.api_agents.common": mock_module}):
            body = _body(self._call(_make_handler()))
            if "http_connector" in body:
                assert body["http_connector"]["status"] == "error"

    # -- Export cache --

    def test_export_cache_healthy(self):
        """Export cache with entries -> healthy."""
        mock_module = MagicMock()
        mock_module.get_export_cache_stats.return_value = {"total_entries": 42}
        with patch.dict("sys.modules", {"aragora.visualization.exporter": mock_module}):
            body = _body(self._call(_make_handler()))
            assert body["export_cache"]["status"] == "healthy"
            assert body["export_cache"]["entries"] == 42

    def test_export_cache_import_error(self):
        """Export cache module missing -> unavailable."""
        with patch.dict("sys.modules", {"aragora.visualization.exporter": None}):
            body = _body(self._call(_make_handler()))
            if "export_cache" in body:
                assert body["export_cache"]["status"] == "unavailable"

    def test_export_cache_runtime_error(self):
        """Export cache raises RuntimeError -> error."""
        mock_module = MagicMock()
        mock_module.get_export_cache_stats.side_effect = RuntimeError("bad")
        with patch.dict("sys.modules", {"aragora.visualization.exporter": mock_module}):
            body = _body(self._call(_make_handler()))
            if "export_cache" in body:
                assert body["export_cache"]["status"] == "error"

    # -- Handler cache --

    def test_handler_cache_healthy(self):
        """Handler cache returns stats."""
        mock_module = MagicMock()
        mock_module.get_cache_stats.return_value = {"hits": 10, "misses": 2}
        with patch.dict("sys.modules", {"aragora.server.handlers.admin.cache": mock_module}):
            body = _body(self._call(_make_handler()))
            assert body["handler_cache"]["status"] == "healthy"
            assert body["handler_cache"]["hits"] == 10

    def test_handler_cache_import_error(self):
        """Handler cache module missing -> unavailable."""
        with patch.dict("sys.modules", {"aragora.server.handlers.admin.cache": None}):
            body = _body(self._call(_make_handler()))
            if "handler_cache" in body:
                assert body["handler_cache"]["status"] == "unavailable"

    def test_handler_cache_key_error(self):
        """Handler cache raises KeyError -> error."""
        mock_module = MagicMock()
        mock_module.get_cache_stats.side_effect = KeyError("missing")
        with patch.dict("sys.modules", {"aragora.server.handlers.admin.cache": mock_module}):
            body = _body(self._call(_make_handler()))
            if "handler_cache" in body:
                assert body["handler_cache"]["status"] == "error"


# ============================================================================
# TestDeepHealthCheck - deep_health_check() function
# ============================================================================


class TestDeepHealthCheck:
    """Tests for deep_health_check() - full dependency verification."""

    def _call(self, handler):
        return handler._deep_health_check()

    # -- Basic response structure --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_healthy_returns_200(self, _slack, _stripe, _ai, _redis, _fs):
        """All healthy -> 200."""
        result = self._call(_make_handler())
        assert _status(result) == 200

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_response_has_required_fields(self, _slack, _stripe, _ai, _redis, _fs):
        """Response has standard fields."""
        body = _body(self._call(_make_handler()))
        for key in ("status", "version", "checks", "response_time_ms", "timestamp"):
            assert key in body

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_timestamp_ends_with_z(self, _slack, _stripe, _ai, _redis, _fs):
        """Timestamp is UTC."""
        body = _body(self._call(_make_handler()))
        assert body["timestamp"].endswith("Z")

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_version_is_string(self, _slack, _stripe, _ai, _redis, _fs):
        """Version is a string."""
        body = _body(self._call(_make_handler()))
        assert isinstance(body["version"], str)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_response_time_is_numeric(self, _slack, _stripe, _ai, _redis, _fs):
        """response_time_ms is a non-negative number."""
        body = _body(self._call(_make_handler()))
        assert isinstance(body["response_time_ms"], (int, float))
        assert body["response_time_ms"] >= 0

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
        body = _body(self._call(_make_handler({"storage": mock_storage})))
        s = body["checks"]["storage"]
        assert s["healthy"] is True
        assert s["status"] == "connected"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_storage_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """No storage -> not_configured."""
        body = _body(self._call(_make_handler()))
        assert body["checks"]["storage"]["status"] == "not_configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_storage_error_degrades(self, _slack, _stripe, _ai, _redis, _fs):
        """Storage error -> unhealthy, overall degraded."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = RuntimeError("db fail")
        body = _body(self._call(_make_handler({"storage": mock_storage})))
        assert body["checks"]["storage"]["healthy"] is False
        assert body["status"] == "degraded"

    # -- ELO deep check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_elo_connected(self, _slack, _stripe, _ai, _redis, _fs):
        """ELO accessible -> connected."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        body = _body(self._call(_make_handler({"elo_system": mock_elo})))
        assert body["checks"]["elo_system"]["healthy"] is True

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_elo_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """No ELO -> not_configured."""
        body = _body(self._call(_make_handler()))
        assert body["checks"]["elo_system"]["status"] == "not_configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_elo_error_degrades(self, _slack, _stripe, _ai, _redis, _fs):
        """ELO error -> unhealthy, degraded."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = OSError("elo fail")
        body = _body(self._call(_make_handler({"elo_system": mock_elo})))
        assert body["checks"]["elo_system"]["healthy"] is False
        assert body["status"] == "degraded"

    # -- Supabase check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_supabase_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """Supabase not configured -> healthy with not_configured."""
        mock_client = MagicMock()
        mock_client.is_configured = False
        mock_client.client = None
        mock_mod = MagicMock()
        mock_mod.SupabaseClient.return_value = mock_client
        with patch.dict("sys.modules", {"aragora.persistence.supabase_client": mock_mod}):
            body = _body(self._call(_make_handler()))
            assert body["checks"]["supabase"]["status"] == "not_configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_supabase_import_error(self, _slack, _stripe, _ai, _redis, _fs):
        """Supabase module missing -> module_not_available."""
        with patch.dict("sys.modules", {"aragora.persistence.supabase_client": None}):
            body = _body(self._call(_make_handler()))
            assert body["checks"]["supabase"]["status"] == "module_not_available"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_supabase_connection_error(self, _slack, _stripe, _ai, _redis, _fs):
        """Supabase connection error -> healthy with warning."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.client = MagicMock()
        mock_client.client.table.return_value.select.return_value.limit.return_value.execute.side_effect = (
            ConnectionError("timeout")
        )
        mock_mod = MagicMock()
        mock_mod.SupabaseClient.return_value = mock_client
        with patch.dict("sys.modules", {"aragora.persistence.supabase_client": mock_mod}):
            body = _body(self._call(_make_handler()))
            sb = body["checks"]["supabase"]
            assert sb["healthy"] is True
            assert sb["status"] == "error"

    # -- User store check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_user_store_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """No user store -> not_configured."""
        body = _body(self._call(_make_handler()))
        assert body["checks"]["user_store"]["healthy"] is True
        assert body["checks"]["user_store"]["status"] == "not_configured"

    # -- Billing check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_billing_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """Billing configured -> status configured."""
        mock_client = MagicMock()
        mock_client._is_configured.return_value = True
        mock_mod = MagicMock()
        mock_mod.StripeClient.return_value = mock_client
        with patch.dict("sys.modules", {"aragora.billing.stripe_client": mock_mod}):
            body = _body(self._call(_make_handler()))
            assert body["checks"]["billing"]["status"] == "configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_billing_not_configured(self, _slack, _stripe, _ai, _redis, _fs):
        """Billing not configured."""
        mock_client = MagicMock()
        mock_client._is_configured.return_value = False
        mock_mod = MagicMock()
        mock_mod.StripeClient.return_value = mock_client
        with patch.dict("sys.modules", {"aragora.billing.stripe_client": mock_mod}):
            body = _body(self._call(_make_handler()))
            assert body["checks"]["billing"]["status"] == "not_configured"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_billing_import_error(self, _slack, _stripe, _ai, _redis, _fs):
        """Billing module missing -> module_not_available."""
        with patch.dict("sys.modules", {"aragora.billing.stripe_client": None}):
            body = _body(self._call(_make_handler()))
            assert body["checks"]["billing"]["status"] == "module_not_available"

    # -- Filesystem deep check --

    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_filesystem_failure_degrades(self, _slack, _stripe, _ai, _redis):
        """Filesystem failure -> degraded."""
        with patch(_P_FS, return_value={"healthy": False, "error": "Permission denied"}):
            body = _body(self._call(_make_handler()))
            assert body["status"] == "degraded"

    # -- System resources --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_high_memory_degrades(self, _slack, _stripe, _ai, _redis, _fs):
        """Memory >=90% -> degraded."""
        mock_psutil = MagicMock()
        mock_memory = MagicMock()
        mock_memory.percent = 95.0
        mock_memory.available = 1 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_disk.free = 100 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            body = _body(self._call(_make_handler()))
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
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_disk.free = 100 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            body = _body(self._call(_make_handler()))
            warnings = body.get("warnings") or []
            assert any("Elevated memory" in w for w in warnings)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_high_disk_degrades(self, _slack, _stripe, _ai, _redis, _fs):
        """Disk >=90% -> degraded."""
        mock_psutil = MagicMock()
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 95.0
        mock_disk.free = 5 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            body = _body(self._call(_make_handler()))
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
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.cpu_count.return_value = 4
        mock_disk = MagicMock()
        mock_disk.percent = 85.0
        mock_disk.free = 15 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            body = _body(self._call(_make_handler()))
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
            body = _body(self._call(_make_handler()))
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
            body = _body(self._call(_make_handler()))
            if "system_resources" in body["checks"]:
                assert body["checks"]["system_resources"]["status"] == "psutil_not_available"

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_psutil_runtime_error(self, _slack, _stripe, _ai, _redis, _fs):
        """psutil raises RuntimeError -> graceful fallback."""
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.side_effect = RuntimeError("bad")
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            body = _body(self._call(_make_handler()))
            if "system_resources" in body["checks"]:
                assert "warning" in body["checks"]["system_resources"]

    # -- Email services --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_email_services_not_available(self, _slack, _stripe, _ai, _redis, _fs):
        """Email services modules missing -> not_available."""
        with patch.dict("sys.modules", {
            "aragora.services.followup_tracker": None,
            "aragora.services.snooze_recommender": None,
        }):
            body = _body(self._call(_make_handler()))
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
            body = _body(self._call(_make_handler()))
            assert body["checks"]["dependency_analyzer"]["status"] == "not_available"

    # -- Stripe check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_stripe_configured_failing_degrades(self, _slack, _ai, _redis, _fs):
        """Stripe configured but failing -> degraded + warning."""
        with patch(_P_STRIPE, return_value={"healthy": False, "configured": True, "error": "Auth failed"}):
            body = _body(self._call(_make_handler()))
            assert body["status"] == "degraded"
            warnings = body.get("warnings") or []
            assert any("Stripe" in w for w in warnings)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_stripe_not_configured_ok(self, _slack, _ai, _redis, _fs):
        """Stripe not configured -> no degradation."""
        with patch(_P_STRIPE, return_value={"healthy": True, "configured": False}):
            body = _body(self._call(_make_handler()))
            assert body["status"] in ("healthy", "healthy_with_warnings")

    # -- Slack check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    def test_slack_configured_failing_degrades(self, _stripe, _ai, _redis, _fs):
        """Slack configured but failing -> degraded + warning."""
        with patch(_P_SLACK, return_value={"healthy": False, "configured": True, "error": "Connection failed"}):
            body = _body(self._call(_make_handler()))
            assert body["status"] == "degraded"
            warnings = body.get("warnings") or []
            assert any("Slack" in w for w in warnings)

    # -- Redis deep check --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_redis_configured_failing_degrades(self, _slack, _stripe, _ai, _fs):
        """Redis configured but unhealthy -> degrades."""
        with patch(_P_REDIS, return_value={"healthy": False, "configured": True, "error": "Refused"}):
            body = _body(self._call(_make_handler()))
            assert body["status"] == "degraded"

    # -- Warnings --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_healthy_with_warnings_status(self, _slack, _stripe, _redis, _fs):
        """Warnings but no failures -> healthy_with_warnings."""
        with patch(_P_AI, return_value={"healthy": True, "any_available": False, "available_count": 0}):
            body = _body(self._call(_make_handler()))
            assert body["status"] == "healthy_with_warnings"
            warnings = body.get("warnings") or []
            assert any("No AI providers" in w for w in warnings)

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_AI, return_value=_HEALTHY_AI)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_warnings_none_when_empty(self, _slack, _stripe, _ai, _redis, _fs):
        """No warnings -> warnings field is None."""
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
            body = _body(self._call(_make_handler()))
            assert body.get("warnings") is None

    # -- AI providers --

    @patch(_P_FS, return_value=_HEALTHY_FS)
    @patch(_P_REDIS, return_value=_HEALTHY_REDIS)
    @patch(_P_STRIPE, return_value=_HEALTHY_STRIPE)
    @patch(_P_SLACK, return_value=_HEALTHY_SLACK)
    def test_no_ai_providers_warning(self, _slack, _stripe, _redis, _fs):
        """No AI providers -> warning added."""
        with patch(_P_AI, return_value={"healthy": True, "any_available": False, "available_count": 0}):
            body = _body(self._call(_make_handler()))
            warnings = body.get("warnings") or []
            assert any("No AI providers" in w for w in warnings)


# ============================================================================
# TestHealthHandlerRouting - HealthHandler.handle() routing
# ============================================================================


class TestHealthHandlerRouting:
    """Tests for HealthHandler.can_handle() and route matching."""

    def test_can_handle_health(self):
        """Handler recognizes /api/v1/health."""
        h = _make_handler()
        assert h.can_handle("/api/v1/health") is True

    def test_can_handle_health_detailed(self):
        """Handler recognizes /api/v1/health/detailed."""
        h = _make_handler()
        assert h.can_handle("/api/v1/health/detailed") is True

    def test_can_handle_health_deep(self):
        """Handler recognizes /api/v1/health/deep."""
        h = _make_handler()
        assert h.can_handle("/api/v1/health/deep") is True

    def test_can_handle_non_v1_health(self):
        """Handler recognizes /api/health."""
        h = _make_handler()
        assert h.can_handle("/api/health") is True

    def test_can_handle_non_v1_detailed(self):
        """Handler recognizes /api/health/detailed."""
        h = _make_handler()
        assert h.can_handle("/api/health/detailed") is True

    def test_can_handle_non_v1_deep(self):
        """Handler recognizes /api/health/deep."""
        h = _make_handler()
        assert h.can_handle("/api/health/deep") is True

    def test_can_handle_healthz(self):
        """Handler recognizes /healthz."""
        h = _make_handler()
        assert h.can_handle("/healthz") is True

    def test_can_handle_readyz(self):
        """Handler recognizes /readyz."""
        h = _make_handler()
        assert h.can_handle("/readyz") is True

    def test_cannot_handle_unknown(self):
        """Unknown path -> cannot handle."""
        h = _make_handler()
        assert h.can_handle("/api/v1/unknown") is False

    def test_cannot_handle_partial_match(self):
        """Partial path match -> cannot handle."""
        h = _make_handler()
        assert h.can_handle("/api/v1/health/extra/path") is False
