"""
Tests for Knowledge Mound Health Check Handler.

Tests cover:
- knowledge_mound_health() - Full KM subsystem health check
- decay_health() - Confidence decay scheduler health check
- Component health aggregation
- Response time tracking
- Warning aggregation
- Status determination (healthy, degraded, not_configured, unavailable)
"""

import sys
import types as _types_mod

# Pre-stub Slack modules to avoid circular ImportError
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
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
import pytest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aragora.server.handlers.admin.health.knowledge_mound import (
    knowledge_mound_health,
    decay_health,
)


# =============================================================================
# Helpers
# =============================================================================


def _parse_result(result):
    """Parse HandlerResult into (body_dict, status_code)."""
    if hasattr(result, "body"):
        body = json.loads(result.body) if result.body else {}
    else:
        body = {}
    status = result.status_code if hasattr(result, "status_code") else 200
    return body, status


def _make_healthy_component(name: str, status: str = "active"):
    """Create a healthy component result."""
    return {
        "healthy": True,
        "status": status,
        "message": f"{name} is operational",
    }


def _make_unhealthy_component(name: str, error: str = "Error"):
    """Create an unhealthy component result."""
    return {
        "healthy": False,
        "status": "error",
        "error": error,
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    return MagicMock()


@pytest.fixture
def mock_all_utils_healthy():
    """Mock all utility functions to return healthy status."""
    return {
        "check_knowledge_mound_module": (
            {"healthy": True, "status": "available"},
            False,
        ),
        "check_mound_core_initialization": (
            {"healthy": True, "status": "active"},
            MagicMock(),
        ),
        "check_storage_backend": {"healthy": True, "status": "active", "type": "postgres"},
        "check_culture_accumulator": {"healthy": True, "status": "active", "patterns": 10},
        "check_staleness_tracker": {"healthy": True, "status": "active", "stale_count": 0},
        "check_rlm_integration": {"healthy": True, "status": "active"},
        "check_codebase_context": {"healthy": True, "status": "active"},
        "check_debate_integration": {"healthy": True, "status": "active"},
        "check_knowledge_mound_redis_cache": {"healthy": True, "status": "active"},
        "check_bidirectional_adapters": {"healthy": True, "status": "active", "adapters": 10},
        "check_control_plane_adapter": {"healthy": True, "status": "active"},
        "check_km_metrics": {"healthy": True, "status": "active"},
        "check_confidence_decay_scheduler": (
            {"healthy": True, "status": "active", "running": True},
            [],
        ),
    }


# =============================================================================
# Tests: knowledge_mound_health - Module Not Available
# =============================================================================


class TestKnowledgeMoundHealthModuleUnavailable:
    """Tests for when Knowledge Mound module is not available."""

    def test_module_not_installed(self, mock_handler):
        """Returns unavailable when module is not installed."""
        with patch(
            "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_module"
        ) as mock_check:
            mock_check.return_value = (
                {"healthy": False, "status": "not_installed"},
                True,  # should_abort
            )

            result = knowledge_mound_health(mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "unavailable"
        assert "not installed" in body["error"]

    def test_module_import_error(self, mock_handler):
        """Returns unavailable on import error."""
        with patch(
            "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_module"
        ) as mock_check:
            mock_check.return_value = (
                {"healthy": False, "status": "import_error", "error": "ModuleNotFoundError"},
                True,
            )

            result = knowledge_mound_health(mock_handler)

        body, status = _parse_result(result)
        assert body["status"] == "unavailable"


# =============================================================================
# Tests: knowledge_mound_health - Healthy State
# =============================================================================


class TestKnowledgeMoundHealthHealthy:
    """Tests for healthy Knowledge Mound status."""

    def test_all_components_healthy(self, mock_handler, mock_all_utils_healthy):
        """Returns healthy when all components are healthy."""
        with (
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_module",
                return_value=mock_all_utils_healthy["check_knowledge_mound_module"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_mound_core_initialization",
                return_value=mock_all_utils_healthy["check_mound_core_initialization"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_storage_backend",
                return_value=mock_all_utils_healthy["check_storage_backend"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_culture_accumulator",
                return_value=mock_all_utils_healthy["check_culture_accumulator"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_staleness_tracker",
                return_value=mock_all_utils_healthy["check_staleness_tracker"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_rlm_integration",
                return_value=mock_all_utils_healthy["check_rlm_integration"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_codebase_context",
                return_value=mock_all_utils_healthy["check_codebase_context"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_debate_integration",
                return_value=mock_all_utils_healthy["check_debate_integration"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_redis_cache",
                return_value=mock_all_utils_healthy["check_knowledge_mound_redis_cache"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_bidirectional_adapters",
                return_value=mock_all_utils_healthy["check_bidirectional_adapters"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_control_plane_adapter",
                return_value=mock_all_utils_healthy["check_control_plane_adapter"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_km_metrics",
                return_value=mock_all_utils_healthy["check_km_metrics"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_confidence_decay_scheduler",
                return_value=mock_all_utils_healthy["check_confidence_decay_scheduler"],
            ),
        ):
            result = knowledge_mound_health(mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "healthy"
        assert body["summary"]["healthy"] == body["summary"]["total_components"]
        assert body["response_time_ms"] >= 0
        assert "timestamp" in body

    def test_response_includes_all_components(self, mock_handler, mock_all_utils_healthy):
        """Response includes all expected components."""
        with (
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_module",
                return_value=mock_all_utils_healthy["check_knowledge_mound_module"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_mound_core_initialization",
                return_value=mock_all_utils_healthy["check_mound_core_initialization"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_storage_backend",
                return_value=mock_all_utils_healthy["check_storage_backend"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_culture_accumulator",
                return_value=mock_all_utils_healthy["check_culture_accumulator"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_staleness_tracker",
                return_value=mock_all_utils_healthy["check_staleness_tracker"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_rlm_integration",
                return_value=mock_all_utils_healthy["check_rlm_integration"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_codebase_context",
                return_value=mock_all_utils_healthy["check_codebase_context"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_debate_integration",
                return_value=mock_all_utils_healthy["check_debate_integration"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_redis_cache",
                return_value=mock_all_utils_healthy["check_knowledge_mound_redis_cache"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_bidirectional_adapters",
                return_value=mock_all_utils_healthy["check_bidirectional_adapters"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_control_plane_adapter",
                return_value=mock_all_utils_healthy["check_control_plane_adapter"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_km_metrics",
                return_value=mock_all_utils_healthy["check_km_metrics"],
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_confidence_decay_scheduler",
                return_value=mock_all_utils_healthy["check_confidence_decay_scheduler"],
            ),
        ):
            result = knowledge_mound_health(mock_handler)

        body, _ = _parse_result(result)
        components = body["components"]

        expected_components = [
            "module",
            "core",
            "storage",
            "culture_accumulator",
            "staleness_tracker",
            "rlm_integration",
            "codebase_context",
            "debate_integration",
            "redis_cache",
            "bidirectional_adapters",
            "control_plane_adapter",
            "km_metrics",
            "confidence_decay",
        ]

        for comp in expected_components:
            assert comp in components, f"Missing component: {comp}"


# =============================================================================
# Tests: knowledge_mound_health - Degraded State
# =============================================================================


class TestKnowledgeMoundHealthDegraded:
    """Tests for degraded Knowledge Mound status."""

    def test_core_unhealthy_returns_degraded(self, mock_handler):
        """Returns degraded when core is unhealthy."""
        with (
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_module",
                return_value=({"healthy": True, "status": "available"}, False),
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_mound_core_initialization",
                return_value=(
                    {"healthy": False, "status": "error", "error": "Init failed"},
                    None,
                ),
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_storage_backend",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_culture_accumulator",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_staleness_tracker",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_rlm_integration",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_codebase_context",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_debate_integration",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_redis_cache",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_bidirectional_adapters",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_control_plane_adapter",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_km_metrics",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_confidence_decay_scheduler",
                return_value=({"healthy": True, "status": "active"}, []),
            ),
        ):
            result = knowledge_mound_health(mock_handler)

        body, _ = _parse_result(result)
        assert body["status"] == "degraded"

    def test_warnings_aggregated(self, mock_handler):
        """Warnings from components are aggregated."""
        with (
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_module",
                return_value=({"healthy": True, "status": "available"}, False),
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_mound_core_initialization",
                return_value=({"healthy": True, "status": "active"}, MagicMock()),
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_storage_backend",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_culture_accumulator",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_staleness_tracker",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_rlm_integration",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_codebase_context",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_debate_integration",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_redis_cache",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_bidirectional_adapters",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_control_plane_adapter",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_km_metrics",
                return_value={"healthy": True, "status": "active"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_confidence_decay_scheduler",
                return_value=(
                    {"healthy": True, "status": "active"},
                    ["Workspace stale warning 1", "Workspace stale warning 2"],
                ),
            ),
        ):
            result = knowledge_mound_health(mock_handler)

        body, _ = _parse_result(result)
        assert body["warnings"] is not None
        assert len(body["warnings"]) == 2


# =============================================================================
# Tests: knowledge_mound_health - Not Configured
# =============================================================================


class TestKnowledgeMoundHealthNotConfigured:
    """Tests for not_configured status."""

    def test_no_active_components(self, mock_handler):
        """Returns not_configured when no components are active."""
        with (
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_module",
                return_value=({"healthy": True, "status": "available"}, False),
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_mound_core_initialization",
                return_value=(
                    {"healthy": True, "status": "not_configured"},
                    None,
                ),
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_storage_backend",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_culture_accumulator",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_staleness_tracker",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_rlm_integration",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_codebase_context",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_debate_integration",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_knowledge_mound_redis_cache",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_bidirectional_adapters",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_control_plane_adapter",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_km_metrics",
                return_value={"healthy": True, "status": "not_configured"},
            ),
            patch(
                "aragora.server.handlers.admin.health.knowledge_mound.check_confidence_decay_scheduler",
                return_value=({"healthy": True, "status": "not_configured"}, []),
            ),
        ):
            result = knowledge_mound_health(mock_handler)

        body, _ = _parse_result(result)
        assert body["status"] == "not_configured"
        assert body["summary"]["active"] == 0


# =============================================================================
# Tests: decay_health - Module Not Available
# =============================================================================


class TestDecayHealthModuleUnavailable:
    """Tests for when decay scheduler module is not available."""

    def test_module_not_installed(self, mock_handler):
        """Returns 503 when module is not installed."""
        with patch.dict(
            sys.modules,
            {"aragora.knowledge.mound.confidence_decay_scheduler": None},
        ):
            # Force ImportError by patching the import inside the function
            original_import = (
                __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
            )

            def mock_import(name, *args, **kwargs):
                if name == "aragora.knowledge.mound.confidence_decay_scheduler":
                    raise ImportError("Module not found")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = decay_health(mock_handler)

        body, status = _parse_result(result)
        assert status == 503
        assert body["status"] == "not_available"


# =============================================================================
# Tests: decay_health - Scheduler Not Configured
# =============================================================================


class TestDecayHealthNotConfigured:
    """Tests for when scheduler is not configured."""

    def test_scheduler_not_initialized(self, mock_handler):
        """Returns not_configured when scheduler not initialized."""
        mock_module = MagicMock()
        mock_module.get_decay_scheduler.return_value = None
        mock_module.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            sys.modules,
            {"aragora.knowledge.mound.confidence_decay_scheduler": mock_module},
        ):
            # Re-import to get the patched module
            from aragora.server.handlers.admin.health import knowledge_mound

            result = knowledge_mound.decay_health(mock_handler)

        body, status = _parse_result(result)
        assert body["status"] == "not_configured"
        assert body["metrics_available"] is True


# =============================================================================
# Tests: decay_health - Healthy State
# =============================================================================


class TestDecayHealthHealthy:
    """Tests for healthy decay scheduler status."""

    def test_scheduler_running_healthy(self, mock_handler):
        """Returns healthy when scheduler is running with no stale workspaces."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "min_confidence_threshold": 0.1,
            "decay_rate": 0.95,
            "total_decay_cycles": 100,
            "total_items_decayed": 500,
            "total_items_expired": 10,
            "decay_errors": 0,
            "last_decay_per_workspace": {
                "ws_1": datetime.now(timezone.utc).isoformat() + "Z",
                "ws_2": (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat() + "Z",
            },
        }

        mock_module = MagicMock()
        mock_module.get_decay_scheduler.return_value = mock_scheduler
        mock_module.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            sys.modules,
            {"aragora.knowledge.mound.confidence_decay_scheduler": mock_module},
        ):
            from aragora.server.handlers.admin.health import knowledge_mound

            result = knowledge_mound.decay_health(mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "healthy"
        assert body["scheduler"]["running"] is True
        assert body["statistics"]["total_cycles"] == 100
        assert body["workspaces"]["stale_count"] == 0


# =============================================================================
# Tests: decay_health - Degraded State
# =============================================================================


class TestDecayHealthDegraded:
    """Tests for degraded decay scheduler status."""

    def test_stale_workspaces_causes_degraded(self, mock_handler):
        """Returns degraded when workspaces have stale decay."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        # Workspace with decay older than 48 hours
        # Note: isoformat() with timezone-aware datetime already includes +00:00,
        # so we use replace("+00:00", "Z") to get the "Z" suffix format
        stale_time = (
            (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat().replace("+00:00", "Z")
        )
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "last_decay_per_workspace": {
                "ws_stale": stale_time,
            },
        }

        mock_module = MagicMock()
        mock_module.get_decay_scheduler.return_value = mock_scheduler
        mock_module.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            sys.modules,
            {"aragora.knowledge.mound.confidence_decay_scheduler": mock_module},
        ):
            from aragora.server.handlers.admin.health import knowledge_mound

            result = knowledge_mound.decay_health(mock_handler)

        body, status = _parse_result(result)
        assert body["status"] == "degraded"
        assert body["workspaces"]["stale_count"] == 1
        assert len(body["warnings"]) >= 1


# =============================================================================
# Tests: decay_health - Stopped State
# =============================================================================


class TestDecayHealthStopped:
    """Tests for stopped decay scheduler status."""

    def test_scheduler_not_running(self, mock_handler):
        """Returns stopped when scheduler is not running."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = False
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "last_decay_per_workspace": {},
        }

        mock_module = MagicMock()
        mock_module.get_decay_scheduler.return_value = mock_scheduler
        mock_module.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            sys.modules,
            {"aragora.knowledge.mound.confidence_decay_scheduler": mock_module},
        ):
            from aragora.server.handlers.admin.health import knowledge_mound

            result = knowledge_mound.decay_health(mock_handler)

        body, status = _parse_result(result)
        assert body["status"] == "stopped"
        assert body["scheduler"]["running"] is False


# =============================================================================
# Tests: decay_health - Edge Cases
# =============================================================================


class TestDecayHealthEdgeCases:
    """Tests for edge cases in decay health."""

    def test_invalid_timestamp_format(self, mock_handler):
        """Handles invalid timestamp formats gracefully."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "last_decay_per_workspace": {
                "ws_invalid": "not-a-valid-timestamp",
            },
        }

        mock_module = MagicMock()
        mock_module.get_decay_scheduler.return_value = mock_scheduler
        mock_module.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            sys.modules,
            {"aragora.knowledge.mound.confidence_decay_scheduler": mock_module},
        ):
            from aragora.server.handlers.admin.health import knowledge_mound

            result = knowledge_mound.decay_health(mock_handler)

        body, status = _parse_result(result)
        # Should not crash, should handle gracefully
        assert status == 200
        assert body["workspaces"]["details"]["ws_invalid"]["parse_error"] is True

    def test_empty_workspace_list(self, mock_handler):
        """Handles empty workspace list."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "last_decay_per_workspace": {},
        }

        mock_module = MagicMock()
        mock_module.get_decay_scheduler.return_value = mock_scheduler
        mock_module.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            sys.modules,
            {"aragora.knowledge.mound.confidence_decay_scheduler": mock_module},
        ):
            from aragora.server.handlers.admin.health import knowledge_mound

            result = knowledge_mound.decay_health(mock_handler)

        body, status = _parse_result(result)
        assert body["status"] == "healthy"
        assert body["workspaces"]["total"] == 0
        assert body["workspaces"]["details"] is None

    def test_response_time_tracked(self, mock_handler):
        """Response time is tracked in response."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {},
        }

        mock_module = MagicMock()
        mock_module.get_decay_scheduler.return_value = mock_scheduler
        mock_module.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            sys.modules,
            {"aragora.knowledge.mound.confidence_decay_scheduler": mock_module},
        ):
            from aragora.server.handlers.admin.health import knowledge_mound

            result = knowledge_mound.decay_health(mock_handler)

        body, _ = _parse_result(result)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0
        assert "timestamp" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
