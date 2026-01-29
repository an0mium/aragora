"""Tests for health check helper functions."""

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
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class MockHandler:
    """Mock handler for testing helper functions."""

    def __init__(self, ctx: Dict[str, Any] | None = None):
        self.ctx = ctx or {}


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


class TestSyncStatus:
    """Tests for sync_status function."""

    def test_sync_enabled_and_running(self):
        """Test sync status when enabled and running."""
        from aragora.server.handlers.admin.health.helpers import sync_status

        handler = MockHandler()

        mock_status = MagicMock()
        mock_status.enabled = True
        mock_status.running = True
        mock_status.queue_size = 5
        mock_status.synced_count = 100
        mock_status.failed_count = 2
        mock_status.last_sync_at = datetime.now(timezone.utc)
        mock_status.last_error = None

        mock_service = MagicMock()
        mock_service.get_status.return_value = mock_status

        with patch(
            "aragora.server.handlers.admin.health.helpers.get_sync_service",
            return_value=mock_service,
        ):
            result = sync_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["enabled"] is True
        assert body["running"] is True
        assert body["queue_size"] == 5

    def test_sync_not_available(self):
        """Test sync status when module not available."""
        from aragora.server.handlers.admin.health.helpers import sync_status

        handler = MockHandler()

        with patch(
            "aragora.server.handlers.admin.health.helpers.get_sync_service",
            side_effect=ImportError("Module not found"),
        ):
            result = sync_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["enabled"] is False
        assert "not available" in body["error"]


class TestSlowDebatesStatus:
    """Tests for slow_debates_status function."""

    def test_no_slow_debates(self):
        """Test status when no slow debates."""
        from aragora.server.handlers.admin.health.helpers import slow_debates_status

        handler = MockHandler()

        with (
            patch(
                "aragora.server.handlers.admin.health.helpers.get_active_debates",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.admin.health.helpers.get_active_debates_lock",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
        ):
            result = slow_debates_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"
        assert body["current_slow_count"] == 0

    def test_slow_debates_detected(self):
        """Test status when slow debates detected."""
        from aragora.server.handlers.admin.health.helpers import slow_debates_status
        import time

        handler = MockHandler()

        # Create a debate that started 60 seconds ago (above threshold)
        active_debates = {
            "debate-1": {
                "start_time": time.time() - 60,
                "task": "Test debate",
                "agents": ["claude", "gemini"],
                "current_round": 2,
                "total_rounds": 3,
            }
        }

        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.admin.health.helpers.get_active_debates",
                return_value=active_debates,
            ),
            patch(
                "aragora.server.handlers.admin.health.helpers.get_active_debates_lock",
                return_value=mock_lock,
            ),
        ):
            result = slow_debates_status(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "degraded"
        assert body["current_slow_count"] == 1
        assert body["current_slow"][0]["debate_id"] == "debate-1"


class TestCircuitBreakersStatus:
    """Tests for circuit_breakers_status function."""

    def test_all_circuits_closed(self):
        """Test status when all circuits closed."""
        from aragora.server.handlers.admin.health.helpers import circuit_breakers_status

        handler = MockHandler()

        mock_metrics = {
            "health": {"status": "healthy"},
            "summary": {"open_count": 0, "closed_count": 5, "half_open_count": 0},
            "circuit_breakers": {},
        }

        with patch(
            "aragora.server.handlers.admin.health.helpers.get_circuit_breaker_metrics",
            return_value=mock_metrics,
        ):
            result = circuit_breakers_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"
        assert body["summary"]["open_count"] == 0

    def test_some_circuits_open(self):
        """Test status when some circuits open."""
        from aragora.server.handlers.admin.health.helpers import circuit_breakers_status

        handler = MockHandler()

        mock_metrics = {
            "health": {"status": "degraded"},
            "summary": {"open_count": 2, "closed_count": 3, "half_open_count": 0},
            "circuit_breakers": {
                "anthropic": {"state": "open"},
                "openai": {"state": "open"},
            },
        }

        with patch(
            "aragora.server.handlers.admin.health.helpers.get_circuit_breaker_metrics",
            return_value=mock_metrics,
        ):
            result = circuit_breakers_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "degraded"
        assert body["summary"]["open_count"] == 2

    def test_module_not_available(self):
        """Test status when module not available."""
        from aragora.server.handlers.admin.health.helpers import circuit_breakers_status

        handler = MockHandler()

        with patch(
            "aragora.server.handlers.admin.health.helpers.get_circuit_breaker_metrics",
            side_effect=ImportError("Module not found"),
        ):
            result = circuit_breakers_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "unavailable"


class TestComponentHealthStatus:
    """Tests for component_health_status function."""

    def test_all_components_healthy(self):
        """Test status when all components healthy."""
        from aragora.server.handlers.admin.health.helpers import component_health_status

        handler = MockHandler()

        mock_component_status = MagicMock()
        mock_component_status.healthy = True
        mock_component_status.consecutive_failures = 0
        mock_component_status.last_error = None
        mock_component_status.latency_ms = 5.0
        mock_component_status.last_check = datetime.now(timezone.utc)
        mock_component_status.metadata = {}

        mock_report = MagicMock()
        mock_report.overall_healthy = True
        mock_report.summary = {"healthy": 3, "unhealthy": 0}
        mock_report.components = {"db": mock_component_status}
        mock_report.checked_at = datetime.now(timezone.utc)

        mock_registry = MagicMock()
        mock_registry.get_report.return_value = mock_report

        with patch(
            "aragora.server.handlers.admin.health.helpers.get_global_health_registry",
            return_value=mock_registry,
        ):
            result = component_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"
        assert body["overall_healthy"] is True

    def test_some_components_unhealthy(self):
        """Test status when some components unhealthy."""
        from aragora.server.handlers.admin.health.helpers import component_health_status

        handler = MockHandler()

        mock_component_status = MagicMock()
        mock_component_status.healthy = False
        mock_component_status.consecutive_failures = 5
        mock_component_status.last_error = "Connection refused"
        mock_component_status.latency_ms = None
        mock_component_status.last_check = datetime.now(timezone.utc)
        mock_component_status.metadata = {}

        mock_report = MagicMock()
        mock_report.overall_healthy = False
        mock_report.summary = {"healthy": 2, "unhealthy": 1}
        mock_report.components = {"redis": mock_component_status}
        mock_report.checked_at = datetime.now(timezone.utc)

        mock_registry = MagicMock()
        mock_registry.get_report.return_value = mock_report

        with patch(
            "aragora.server.handlers.admin.health.helpers.get_global_health_registry",
            return_value=mock_registry,
        ):
            result = component_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "degraded"

    def test_module_not_available(self):
        """Test status when module not available."""
        from aragora.server.handlers.admin.health.helpers import component_health_status

        handler = MockHandler()

        with patch(
            "aragora.server.handlers.admin.health.helpers.get_global_health_registry",
            side_effect=ImportError("Module not found"),
        ):
            result = component_health_status(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "unavailable"
