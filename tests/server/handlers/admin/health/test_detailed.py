"""Tests for detailed health check implementations."""

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
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class MockHandler:
    """Mock handler for testing detailed health functions."""

    def __init__(
        self,
        storage: Any = None,
        elo_system: Any = None,
        nomic_dir: Path | None = None,
        ctx: Dict[str, Any] | None = None,
    ):
        self._storage = storage
        self._elo_system = elo_system
        self._nomic_dir = nomic_dir
        self.ctx = ctx or {}

    def get_storage(self) -> Any:
        return self._storage

    def get_elo_system(self) -> Any:
        return self._elo_system

    def get_nomic_dir(self) -> Path | None:
        return self._nomic_dir


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    # Reset the _SERVER_START_TIME if needed
    import aragora.server.handlers.admin.health as health_mod

    health_mod._HEALTH_CACHE.clear()
    health_mod._HEALTH_CACHE_TIMESTAMPS.clear()
    yield


class TestHealthCheck:
    """Tests for health_check function."""

    def test_health_check_all_healthy(self, tmp_path):
        """Test health check returns healthy when all checks pass."""
        from aragora.server.handlers.admin.health.detailed import health_check

        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = []

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []

        handler = MockHandler(
            storage=mock_storage,
            elo_system=mock_elo,
            nomic_dir=tmp_path,
        )

        with (
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    )
                },
            ),
        ):
            result = health_check(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"
        assert "checks" in body
        assert "version" in body

    def test_health_check_degraded_mode(self):
        """Test health check returns 503 in degraded mode."""
        from aragora.server.handlers.admin.health.detailed import health_check

        handler = MockHandler()

        mock_state = MagicMock()
        mock_state.reason = "Missing API key"
        mock_state.error_code.value = "MISSING_KEY"
        mock_state.recovery_hint = "Set ANTHROPIC_API_KEY"
        mock_state.timestamp = "2024-01-01T00:00:00Z"

        mock_degraded = MagicMock()
        mock_degraded.is_degraded.return_value = True
        mock_degraded.get_degraded_state.return_value = mock_state

        with patch.dict("sys.modules", {"aragora.server.degraded_mode": mock_degraded}):
            with (
                patch(
                    "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                    return_value={"healthy": True},
                ),
                patch(
                    "aragora.server.handlers.admin.health_utils.check_redis_health",
                    return_value={"healthy": True, "configured": False},
                ),
                patch(
                    "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                    return_value={"healthy": True, "any_available": True},
                ),
                patch(
                    "aragora.server.handlers.admin.health.detailed.check_security_services",
                    return_value={"healthy": True},
                ),
            ):
                result = health_check(handler)

        assert result.status_code == 503
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "degraded"

    def test_health_check_filesystem_failure(self, tmp_path):
        """Test health check reports filesystem failure."""
        from aragora.server.handlers.admin.health.detailed import health_check

        handler = MockHandler(nomic_dir=tmp_path)

        with (
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": False, "error": "Permission denied"},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    )
                },
            ),
        ):
            result = health_check(handler)

        assert result.status_code == 503
        body = json.loads(result.body.decode("utf-8"))
        assert body["checks"]["filesystem"]["healthy"] is False

    def test_health_check_circuit_breakers_open(self, tmp_path):
        """Test health check reports degraded when circuit breakers are open."""
        from aragora.server.handlers.admin.health.detailed import health_check

        handler = MockHandler(nomic_dir=tmp_path)

        mock_metrics = {"summary": {"open_count": 5, "half_open_count": 0, "closed_count": 10}}

        with (
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.get_circuit_breaker_metrics",
                return_value=mock_metrics,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    )
                },
            ),
        ):
            result = health_check(handler)

        assert result.status_code == 503
        body = json.loads(result.body.decode("utf-8"))
        assert body["checks"]["circuit_breakers"]["open"] == 5

    def test_health_check_response_time(self, tmp_path):
        """Test health check includes response time."""
        from aragora.server.handlers.admin.health.detailed import health_check

        handler = MockHandler(nomic_dir=tmp_path)

        with (
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    )
                },
            ),
        ):
            result = health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0


class TestWebsocketHealth:
    """Tests for websocket_health function."""

    def test_websocket_not_configured(self):
        """Test websocket health when not configured."""
        from aragora.server.handlers.admin.health.detailed import websocket_health

        handler = MockHandler(ctx={})

        result = websocket_health(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "unavailable"
        assert body["clients"] == 0

    def test_websocket_healthy(self):
        """Test websocket health when manager is healthy."""
        from aragora.server.handlers.admin.health.detailed import websocket_health

        mock_manager = MagicMock()
        mock_manager.clients = ["client1", "client2"]

        handler = MockHandler(ctx={"ws_manager": mock_manager})

        result = websocket_health(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"
        assert body["clients"] == 2

    def test_websocket_error(self):
        """Test websocket health when manager throws error."""
        from aragora.server.handlers.admin.health.detailed import websocket_health

        mock_manager = MagicMock()
        type(mock_manager).clients = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Error"))
        )

        handler = MockHandler(ctx={"ws_manager": mock_manager})

        result = websocket_health(handler)

        assert result.status_code == 503
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "error"


class TestDetailedHealthCheck:
    """Tests for detailed_health_check function."""

    def test_detailed_health_includes_components(self, tmp_path):
        """Test detailed health includes component status."""
        from aragora.server.handlers.admin.health.detailed import detailed_health_check

        mock_storage = MagicMock()
        handler = MockHandler(storage=mock_storage, nomic_dir=tmp_path)

        result = detailed_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "components" in body
        assert body["components"]["storage"] is True

    def test_detailed_health_includes_warnings(self, tmp_path):
        """Test detailed health includes warnings list."""
        from aragora.server.handlers.admin.health.detailed import detailed_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        result = detailed_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "warnings" in body
        assert isinstance(body["warnings"], list)

    def test_detailed_health_observer_metrics(self, tmp_path):
        """Test detailed health includes observer metrics."""
        from aragora.server.handlers.admin.health.detailed import detailed_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {
            "total_calls": 100,
            "failures": 5,
            "failure_rate": 0.05,
        }

        with patch(
            "aragora.server.handlers.admin.health.detailed.SimpleObserver",
            return_value=mock_observer,
        ):
            result = detailed_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "observer" in body
        assert body["observer"]["failure_rate"] == 0.05

    def test_detailed_health_high_failure_rate_degrades(self, tmp_path):
        """Test detailed health reports degraded on high failure rate."""
        from aragora.server.handlers.admin.health.detailed import detailed_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        mock_observer = MagicMock()
        mock_observer.get_report.return_value = {
            "total_calls": 100,
            "failures": 60,
            "failure_rate": 0.6,
        }

        with patch(
            "aragora.server.handlers.admin.health.detailed.SimpleObserver",
            return_value=mock_observer,
        ):
            result = detailed_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "degraded"

    def test_detailed_health_memory_stats(self, tmp_path):
        """Test detailed health includes memory stats when psutil available."""
        from aragora.server.handlers.admin.health.detailed import detailed_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.memory_percent.return_value = 5.0

        with patch("psutil.Process", return_value=mock_process):
            result = detailed_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "memory" in body
        assert body["memory"]["rss_mb"] > 0

    def test_detailed_health_sqlite_production_warning(self, tmp_path):
        """Test detailed health warns about SQLite in production."""
        from aragora.server.handlers.admin.health.detailed import detailed_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        with patch.dict("os.environ", {"ARAGORA_ENV": "production", "DATABASE_URL": ""}):
            result = detailed_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert any("SQLite" in w for w in body.get("warnings", []))


class TestDeepHealthCheck:
    """Tests for deep_health_check function."""

    def test_deep_health_includes_all_dependencies(self, tmp_path):
        """Test deep health checks all external dependencies."""
        from aragora.server.handlers.admin.health.detailed import deep_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        with (
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_stripe_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_slack_health",
                return_value={"healthy": True, "configured": False},
            ),
        ):
            result = deep_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "checks" in body
        assert "storage" in body["checks"]
        assert "elo_system" in body["checks"]
        assert "filesystem" in body["checks"]
        assert "redis" in body["checks"]
        assert "ai_providers" in body["checks"]

    def test_deep_health_system_resources(self, tmp_path):
        """Test deep health checks system resources."""
        from aragora.server.handlers.admin.health.detailed import deep_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024**3

        mock_disk = MagicMock()
        mock_disk.percent = 60.0
        mock_disk.free = 100 * 1024**3

        with (
            patch("psutil.virtual_memory", return_value=mock_memory),
            patch("psutil.cpu_percent", return_value=30.0),
            patch("psutil.cpu_count", return_value=8),
            patch("psutil.disk_usage", return_value=mock_disk),
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_stripe_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_slack_health",
                return_value={"healthy": True, "configured": False},
            ),
        ):
            result = deep_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "memory" in body["checks"]
        assert body["checks"]["memory"]["healthy"] is True
        assert "cpu" in body["checks"]
        assert "disk" in body["checks"]

    def test_deep_health_high_memory_usage(self, tmp_path):
        """Test deep health warns on high memory usage."""
        from aragora.server.handlers.admin.health.detailed import deep_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        mock_memory = MagicMock()
        mock_memory.percent = 92.0
        mock_memory.available = 1 * 1024**3

        with (
            patch("psutil.virtual_memory", return_value=mock_memory),
            patch("psutil.cpu_percent", return_value=30.0),
            patch("psutil.cpu_count", return_value=8),
            patch("psutil.disk_usage", return_value=MagicMock(percent=50.0, free=100 * 1024**3)),
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_stripe_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_slack_health",
                return_value={"healthy": True, "configured": False},
            ),
        ):
            result = deep_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["checks"]["memory"]["healthy"] is False
        assert any("memory" in w.lower() for w in body.get("warnings", []))

    def test_deep_health_low_disk_space(self, tmp_path):
        """Test deep health warns on low disk space."""
        from aragora.server.handlers.admin.health.detailed import deep_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        mock_disk = MagicMock()
        mock_disk.percent = 95.0
        mock_disk.free = 10 * 1024**3

        with (
            patch(
                "psutil.virtual_memory", return_value=MagicMock(percent=50.0, available=8 * 1024**3)
            ),
            patch("psutil.cpu_percent", return_value=30.0),
            patch("psutil.cpu_count", return_value=8),
            patch("psutil.disk_usage", return_value=mock_disk),
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_stripe_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_slack_health",
                return_value={"healthy": True, "configured": False},
            ),
        ):
            result = deep_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["checks"]["disk"]["healthy"] is False

    def test_deep_health_timestamp_included(self, tmp_path):
        """Test deep health includes timestamp."""
        from aragora.server.handlers.admin.health.detailed import deep_health_check

        handler = MockHandler(nomic_dir=tmp_path)

        with (
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_stripe_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_slack_health",
                return_value={"healthy": True, "configured": False},
            ),
        ):
            result = deep_health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")
