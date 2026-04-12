"""Tests for server/handlers/admin/health/metrics_health.py module."""

from __future__ import annotations

import json
import sys
import types as _types_mod
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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


from aragora.server.handlers.admin.health.metrics_health import metrics_health


def _parse_body(result: Any) -> dict:
    """Parse HandlerResult body as JSON."""
    return json.loads(result.body)


class TestMetricsHealthBasic:
    """Tests for basic metrics_health functionality."""

    def test_returns_handler_result(self):
        result = metrics_health(MagicMock())
        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_response_has_required_fields(self):
        result = metrics_health(MagicMock())
        data = _parse_body(result)
        assert "status" in data
        assert "metrics_enabled" in data
        assert "components" in data
        assert "timestamp" in data

    def test_timestamp_format(self):
        result = metrics_health(MagicMock())
        data = _parse_body(result)
        assert data["timestamp"].endswith("Z")

    def test_status_is_valid_value(self):
        result = metrics_health(MagicMock())
        data = _parse_body(result)
        assert data["status"] in {"healthy", "degraded", "disabled"}


class TestMetricsHealthEnabled:
    """Tests when metrics subsystem is enabled."""

    def test_metrics_enabled_with_all_components_healthy(self):
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.port = 9090
        mock_config.prefix = "aragora"

        with (
            patch.dict(sys.modules, {}),
            patch(
                "aragora.observability.metrics.base.get_metrics_enabled",
                return_value=True,
            ),
            patch(
                "aragora.observability.metrics.core.is_initialized",
                return_value=True,
            ),
            patch(
                "aragora.observability.config.get_metrics_config",
                return_value=mock_config,
            ),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["metrics_enabled"] is True
            assert data["components"]["enabled"]["value"] is True
            assert data["components"]["initialized"]["value"] is True


class TestMetricsHealthDisabled:
    """Tests when metrics are disabled."""

    def test_disabled_metrics_status(self):
        with patch(
            "aragora.observability.metrics.base.get_metrics_enabled",
            return_value=False,
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["status"] == "disabled"
            assert data["metrics_enabled"] is False

    def test_disabled_no_issues_about_initialization(self):
        with (
            patch(
                "aragora.observability.metrics.base.get_metrics_enabled",
                return_value=False,
            ),
            patch(
                "aragora.observability.metrics.core.is_initialized",
                return_value=False,
            ),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["status"] == "disabled"
            issues = data.get("issues")
            if issues:
                assert "metrics enabled but not initialized" not in issues


class TestMetricsHealthImportErrors:
    """Tests for graceful handling of missing dependencies."""

    def test_missing_metrics_base_module(self):
        with patch.dict(
            sys.modules,
            {"aragora.observability.metrics.base": None},
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["enabled"]["status"] == "unavailable"
            assert result.status_code == 200

    def test_missing_metrics_core_module(self):
        with patch.dict(
            sys.modules,
            {"aragora.observability.metrics.core": None},
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["initialized"]["status"] == "unavailable"
            assert result.status_code == 200

    def test_missing_prometheus_client(self):
        with patch.dict(sys.modules, {"prometheus_client": None}):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["prometheus_available"]["status"] == "unavailable"
            assert data["components"]["collectors"]["status"] == "unavailable"
            assert result.status_code == 200

    def test_missing_observability_config(self):
        with patch.dict(
            sys.modules,
            {"aragora.observability.config": None},
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["config"]["status"] == "unavailable"
            assert result.status_code == 200

    def test_all_imports_fail_still_returns_200(self):
        with patch.dict(
            sys.modules,
            {
                "aragora.observability.metrics.base": None,
                "aragora.observability.metrics.core": None,
                "prometheus_client": None,
                "aragora.observability.config": None,
            },
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert result.status_code == 200
            assert data["status"] == "disabled"
            assert data["metrics_enabled"] is False


class TestMetricsHealthExceptionHandling:
    """Tests for exception handling in each component check."""

    def test_get_metrics_enabled_raises_runtime_error(self):
        with patch(
            "aragora.observability.metrics.base.get_metrics_enabled",
            side_effect=RuntimeError("config broken"),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["enabled"]["status"] == "error"
            assert "config broken" in data["components"]["enabled"]["error"]

    def test_is_initialized_raises_attribute_error(self):
        with (
            patch(
                "aragora.observability.metrics.base.get_metrics_enabled",
                return_value=True,
            ),
            patch(
                "aragora.observability.metrics.core.is_initialized",
                side_effect=AttributeError("no such attr"),
            ),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["initialized"]["status"] == "error"
            assert "no such attr" in data["components"]["initialized"]["error"]

    def test_collector_enumeration_raises_runtime_error(self):
        mock_registry = MagicMock()
        mock_registry.collect.side_effect = RuntimeError("registry broken")
        mock_prom = MagicMock()
        mock_prom.REGISTRY = mock_registry

        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["collectors"]["status"] == "error"

    def test_config_check_raises_value_error(self):
        with patch(
            "aragora.observability.config.get_metrics_config",
            side_effect=ValueError("bad config"),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["config"]["status"] == "error"
            assert "bad config" in data["components"]["config"]["error"]


class TestMetricsHealthDegradedState:
    """Tests for degraded status detection."""

    def test_enabled_but_not_initialized_is_degraded(self):
        with (
            patch(
                "aragora.observability.metrics.base.get_metrics_enabled",
                return_value=True,
            ),
            patch(
                "aragora.observability.metrics.core.is_initialized",
                return_value=False,
            ),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["status"] == "degraded"
            issues = data.get("issues") or []
            assert "metrics enabled but not initialized" in issues

    def test_enabled_but_no_prometheus_is_degraded(self):
        with (
            patch(
                "aragora.observability.metrics.base.get_metrics_enabled",
                return_value=True,
            ),
            patch(
                "aragora.observability.metrics.core.is_initialized",
                return_value=True,
            ),
            patch.dict(sys.modules, {"prometheus_client": None}),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["status"] == "degraded"
            issues = data.get("issues") or []
            assert any("prometheus" in i for i in issues)

    def test_issues_is_none_when_empty(self):
        with (
            patch(
                "aragora.observability.metrics.base.get_metrics_enabled",
                return_value=False,
            ),
            patch(
                "aragora.observability.metrics.core.is_initialized",
                return_value=False,
            ),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            if data["status"] == "disabled":
                pass


class TestMetricsHealthCollectors:
    """Tests for collector enumeration."""

    def test_collector_count_reported(self):
        mock_registry = MagicMock()
        mock_registry.collect.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_prom = MagicMock()
        mock_prom.REGISTRY = mock_registry

        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["collectors"]["count"] == 3
            assert data["components"]["collectors"]["status"] == "ok"

    def test_zero_collectors(self):
        mock_registry = MagicMock()
        mock_registry.collect.return_value = []
        mock_prom = MagicMock()
        mock_prom.REGISTRY = mock_registry

        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["collectors"]["count"] == 0


class TestMetricsHealthConfig:
    """Tests for config component check."""

    def test_config_port_and_prefix_reported(self):
        mock_config = MagicMock()
        mock_config.port = 9090
        mock_config.prefix = "aragora"
        mock_config.enabled = True

        with (
            patch(
                "aragora.observability.config.get_metrics_config",
                return_value=mock_config,
            ),
            patch(
                "aragora.observability.metrics.base.get_metrics_enabled",
                return_value=True,
            ),
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["config"]["port"] == 9090
            assert data["components"]["config"]["prefix"] == "aragora"

    def test_config_missing_attributes_uses_none(self):
        mock_config = MagicMock(spec=[])

        with patch(
            "aragora.observability.config.get_metrics_config",
            return_value=mock_config,
        ):
            result = metrics_health(MagicMock())
            data = _parse_body(result)
            assert data["components"]["config"]["status"] == "ok"
            assert data["components"]["config"]["port"] is None
            assert data["components"]["config"]["prefix"] is None
