"""Tests for integration health check handler.

Covers:
- IntegrationHealthHandler construction and initialization
- can_handle() routing for valid and invalid paths
- handle() dispatch and path filtering
- _get_health() core logic:
  - Integration discovery (all 5 integrations reported)
  - Configured detection via environment variables
  - Module availability checking
  - Connector health and last_check reporting
  - Summary counts (total, configured, healthy)
- Edge cases: empty context, non-dict connectors, missing connector attrs,
  partial env vars, isoformat vs str fallback for last_check
- ROUTES class attribute
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.integrations.health import (
    IntegrationHealthHandler,
    _INTEGRATIONS,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Parse JSON body from HandlerResult."""
    if isinstance(result, dict):
        return result.get("body", result)
    return json.loads(result.body.decode("utf-8"))


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    return result.status_code


def _make_mock_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.headers = {"Content-Length": "2"}
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = json.dumps(body or {}).encode()
    return handler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an IntegrationHealthHandler with no context."""
    return IntegrationHealthHandler()


@pytest.fixture
def handler_with_ctx():
    """Factory for creating handler with custom context."""

    def _make(ctx: dict[str, Any] | None = None) -> IntegrationHealthHandler:
        return IntegrationHealthHandler(ctx=ctx)

    return _make


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler."""
    return _make_mock_handler()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure integration-related env vars are unset by default."""
    env_vars = [
        "SLACK_WEBHOOK_URL",
        "SLACK_BOT_TOKEN",
        "SMTP_HOST",
        "SMTP_SERVER",
        "SENDGRID_API_KEY",
        "DISCORD_WEBHOOK_URL",
        "DISCORD_BOT_TOKEN",
        "TEAMS_WEBHOOK_URL",
        "MS_TEAMS_WEBHOOK",
        "ZAPIER_WEBHOOK_URL",
        "ZAPIER_API_KEY",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Class attribute tests
# ---------------------------------------------------------------------------


class TestHandlerAttributes:
    """Tests for handler class attributes and initialization."""

    def test_routes_contains_health_endpoint(self):
        assert "/api/v1/integrations/health" in IntegrationHealthHandler.ROUTES

    def test_routes_is_list(self):
        assert isinstance(IntegrationHealthHandler.ROUTES, list)

    def test_default_ctx_is_empty_dict(self):
        h = IntegrationHealthHandler()
        assert h.ctx == {}

    def test_ctx_from_constructor(self):
        ctx = {"foo": "bar"}
        h = IntegrationHealthHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_becomes_empty_dict(self):
        h = IntegrationHealthHandler(ctx=None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle() path matching."""

    def test_versioned_health_path(self, handler):
        assert handler.can_handle("/api/v1/integrations/health") is True

    def test_v2_health_path(self, handler):
        assert handler.can_handle("/api/v2/integrations/health") is True

    def test_unversioned_health_path(self, handler):
        assert handler.can_handle("/api/integrations/health") is True

    def test_wrong_path_returns_false(self, handler):
        assert handler.can_handle("/api/v1/integrations/status") is False

    def test_root_path_returns_false(self, handler):
        assert handler.can_handle("/") is False

    def test_empty_string_returns_false(self, handler):
        assert handler.can_handle("") is False

    def test_partial_path_returns_false(self, handler):
        assert handler.can_handle("/api/v1/integrations") is False

    def test_trailing_slash_returns_false(self, handler):
        assert handler.can_handle("/api/v1/integrations/health/") is False

    def test_extra_segment_returns_false(self, handler):
        assert handler.can_handle("/api/v1/integrations/health/detail") is False


# ---------------------------------------------------------------------------
# handle() dispatch tests
# ---------------------------------------------------------------------------


class TestHandle:
    """Tests for handle() dispatch logic."""

    def test_returns_result_for_valid_path(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    def test_returns_none_for_wrong_path(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/other", {}, mock_http)
        assert result is None

    def test_query_params_are_ignored(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {"foo": "bar"}, mock_http)
        assert result is not None
        assert _status(result) == 200

    def test_unversioned_path_dispatches(self, handler, mock_http):
        result = handler.handle("/api/integrations/health", {}, mock_http)
        assert result is not None
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Integration discovery tests
# ---------------------------------------------------------------------------


class TestIntegrationDiscovery:
    """Tests for _INTEGRATIONS constant and discovery."""

    def test_five_integrations_defined(self):
        assert len(_INTEGRATIONS) == 5

    def test_integration_names(self):
        names = [i["name"] for i in _INTEGRATIONS]
        assert names == ["slack", "email", "discord", "teams", "zapier"]

    def test_each_integration_has_env_vars(self):
        for integration in _INTEGRATIONS:
            assert "env_vars" in integration
            assert len(integration["env_vars"]) > 0

    def test_each_integration_has_module(self):
        for integration in _INTEGRATIONS:
            assert "module" in integration
            assert integration["module"].startswith("aragora.integrations.")


# ---------------------------------------------------------------------------
# _get_health: response structure tests
# ---------------------------------------------------------------------------


class TestHealthResponseStructure:
    """Tests for the response structure returned by _get_health."""

    def test_response_has_integrations_key(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert "integrations" in body

    def test_response_has_total_key(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert "total" in body

    def test_response_has_configured_key(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert "configured" in body

    def test_response_has_healthy_key(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert "healthy" in body

    def test_total_equals_five(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["total"] == 5

    def test_integration_entry_has_required_keys(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        for entry in body["integrations"]:
            assert "name" in entry
            assert "configured" in entry
            assert "module_available" in entry
            assert "healthy" in entry
            assert "last_check" in entry


# ---------------------------------------------------------------------------
# _get_health: env var configuration detection
# ---------------------------------------------------------------------------


class TestConfiguredDetection:
    """Tests for integration configured status via env vars."""

    def test_no_env_vars_none_configured(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["configured"] == 0
        for entry in body["integrations"]:
            assert entry["configured"] is False

    def test_slack_webhook_url_configures_slack(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/xxx")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        slack = next(e for e in body["integrations"] if e["name"] == "slack")
        assert slack["configured"] is True
        assert body["configured"] == 1

    def test_slack_bot_token_configures_slack(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        slack = next(e for e in body["integrations"] if e["name"] == "slack")
        assert slack["configured"] is True

    def test_smtp_host_configures_email(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        email = next(e for e in body["integrations"] if e["name"] == "email")
        assert email["configured"] is True

    def test_sendgrid_api_key_configures_email(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("SENDGRID_API_KEY", "SG.xxx")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        email = next(e for e in body["integrations"] if e["name"] == "email")
        assert email["configured"] is True

    def test_discord_webhook_configures_discord(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/xxx")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        discord = next(e for e in body["integrations"] if e["name"] == "discord")
        assert discord["configured"] is True

    def test_teams_webhook_configures_teams(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("TEAMS_WEBHOOK_URL", "https://teams.webhook.office.com/xxx")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        teams = next(e for e in body["integrations"] if e["name"] == "teams")
        assert teams["configured"] is True

    def test_ms_teams_webhook_configures_teams(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("MS_TEAMS_WEBHOOK", "https://teams.webhook.office.com/xxx")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        teams = next(e for e in body["integrations"] if e["name"] == "teams")
        assert teams["configured"] is True

    def test_zapier_webhook_configures_zapier(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("ZAPIER_WEBHOOK_URL", "https://hooks.zapier.com/xxx")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        zapier = next(e for e in body["integrations"] if e["name"] == "zapier")
        assert zapier["configured"] is True

    def test_all_configured(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/xxx")
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/xxx")
        monkeypatch.setenv("TEAMS_WEBHOOK_URL", "https://teams.webhook.office.com/xxx")
        monkeypatch.setenv("ZAPIER_WEBHOOK_URL", "https://hooks.zapier.com/xxx")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["configured"] == 5

    def test_empty_env_var_not_configured(self, handler, mock_http, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "")
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        slack = next(e for e in body["integrations"] if e["name"] == "slack")
        assert slack["configured"] is False


# ---------------------------------------------------------------------------
# _get_health: module availability
# ---------------------------------------------------------------------------


class TestModuleAvailability:
    """Tests for module_available detection."""

    def test_module_available_when_importable(self, handler, mock_http):
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("aragora.integrations."):
                return MagicMock()
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        for entry in body["integrations"]:
            assert entry["module_available"] is True

    def test_module_unavailable_on_import_error(self, handler, mock_http):
        original_import = __import__

        def failing_import(name, *args, **kwargs):
            if name.startswith("aragora.integrations."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        for entry in body["integrations"]:
            assert entry["module_available"] is False


# ---------------------------------------------------------------------------
# _get_health: connector health status
# ---------------------------------------------------------------------------


class TestConnectorHealth:
    """Tests for connector health checking from context."""

    def test_no_connectors_all_unhealthy(self, mock_http):
        h = IntegrationHealthHandler(ctx={})
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["healthy"] == 0
        for entry in body["integrations"]:
            assert entry["healthy"] is False

    def test_connector_healthy_true(self, mock_http):
        connector = MagicMock()
        connector.healthy = True
        connector.last_check = None
        ctx = {"connectors": {"slack": connector}}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        slack = next(e for e in body["integrations"] if e["name"] == "slack")
        assert slack["healthy"] is True
        assert body["healthy"] == 1

    def test_connector_healthy_false(self, mock_http):
        connector = MagicMock()
        connector.healthy = False
        connector.last_check = None
        ctx = {"connectors": {"slack": connector}}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        slack = next(e for e in body["integrations"] if e["name"] == "slack")
        assert slack["healthy"] is False

    def test_multiple_connectors_healthy(self, mock_http):
        connectors = {}
        for name in ["slack", "email", "discord"]:
            c = MagicMock()
            c.healthy = True
            c.last_check = None
            connectors[name] = c
        ctx = {"connectors": connectors}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["healthy"] == 3

    def test_connector_without_healthy_attr(self, mock_http):
        connector = MagicMock(spec=[])  # no attributes at all
        ctx = {"connectors": {"slack": connector}}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        slack = next(e for e in body["integrations"] if e["name"] == "slack")
        assert slack["healthy"] is False

    def test_connectors_not_dict(self, mock_http):
        """When connectors is not a dict, connector lookup returns None."""
        ctx = {"connectors": ["slack", "email"]}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["healthy"] == 0

    def test_connectors_key_missing(self, mock_http):
        """When context has no connectors key."""
        h = IntegrationHealthHandler(ctx={"other": "stuff"})
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["healthy"] == 0


# ---------------------------------------------------------------------------
# _get_health: last_check handling
# ---------------------------------------------------------------------------


class TestLastCheck:
    """Tests for last_check serialization."""

    def test_last_check_none_when_no_connector(self, mock_http):
        h = IntegrationHealthHandler(ctx={})
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        for entry in body["integrations"]:
            assert entry["last_check"] is None

    def test_last_check_with_datetime_isoformat(self, mock_http):
        dt = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
        connector = MagicMock()
        connector.healthy = True
        connector.last_check = dt
        ctx = {"connectors": {"slack": connector}}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        slack = next(e for e in body["integrations"] if e["name"] == "slack")
        assert slack["last_check"] == "2026-02-23T12:00:00+00:00"

    def test_last_check_with_string_value(self, mock_http):
        """When last_check is a plain string without isoformat()."""
        connector = MagicMock(spec=[])
        connector.healthy = False
        connector.last_check = "2026-02-23T12:00:00Z"
        ctx = {"connectors": {"email": connector}}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        email = next(e for e in body["integrations"] if e["name"] == "email")
        assert email["last_check"] == "2026-02-23T12:00:00Z"

    def test_last_check_none_on_connector(self, mock_http):
        connector = MagicMock()
        connector.healthy = False
        connector.last_check = None
        ctx = {"connectors": {"discord": connector}}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        discord = next(e for e in body["integrations"] if e["name"] == "discord")
        assert discord["last_check"] is None

    def test_last_check_with_numeric_value(self, mock_http):
        """When last_check is a numeric timestamp, falls back to str()."""
        connector = MagicMock(spec=[])
        connector.healthy = False
        connector.last_check = 1708700000
        ctx = {"connectors": {"teams": connector}}
        h = IntegrationHealthHandler(ctx=ctx)
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        teams = next(e for e in body["integrations"] if e["name"] == "teams")
        assert teams["last_check"] == "1708700000"


# ---------------------------------------------------------------------------
# Summary counts
# ---------------------------------------------------------------------------


class TestSummaryCounts:
    """Tests for total/configured/healthy summary counts."""

    def test_zero_counts_no_config(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["total"] == 5
        assert body["configured"] == 0
        assert body["healthy"] == 0

    def test_configured_count_matches(self, mock_http, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SENDGRID_API_KEY", "SG.xxx")
        h = IntegrationHealthHandler()
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["configured"] == 2

    def test_healthy_count_matches(self, mock_http):
        connectors = {}
        for name in ["slack", "discord", "zapier"]:
            c = MagicMock()
            c.healthy = True
            c.last_check = None
            connectors[name] = c
        # email and teams have no connectors
        h = IntegrationHealthHandler(ctx={"connectors": connectors})
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["healthy"] == 3

    def test_healthy_does_not_count_unhealthy_connectors(self, mock_http):
        connector = MagicMock()
        connector.healthy = False
        connector.last_check = None
        h = IntegrationHealthHandler(ctx={"connectors": {"slack": connector}})
        result = h.handle("/api/v1/integrations/health", {}, mock_http)
        body = _body(result)
        assert body["healthy"] == 0


# ---------------------------------------------------------------------------
# Content type and status code
# ---------------------------------------------------------------------------


class TestResponseMeta:
    """Tests for response metadata (status code, content type)."""

    def test_status_code_200(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        assert _status(result) == 200

    def test_content_type_json(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        assert result.content_type == "application/json"

    def test_body_is_bytes(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        assert isinstance(result.body, bytes)

    def test_body_is_valid_json(self, handler, mock_http):
        result = handler.handle("/api/v1/integrations/health", {}, mock_http)
        parsed = json.loads(result.body.decode("utf-8"))
        assert isinstance(parsed, dict)
