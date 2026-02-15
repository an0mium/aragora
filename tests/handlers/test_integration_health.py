"""Tests for integration health check endpoint.

Tests:
- GET /api/v1/integrations/health - health status for all integrations
- Configured detection via env vars
- Healthy status from connectors in context
- Module availability detection
- Unrelated paths return None
"""

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.integrations.health import IntegrationHealthHandler


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture
def handler():
    """Create an IntegrationHealthHandler instance."""
    return IntegrationHealthHandler(ctx={})


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    return MagicMock()


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_health_path(self, handler):
        assert handler.can_handle("/api/v1/integrations/health") is True

    def test_does_not_handle_other_path(self, handler):
        assert handler.can_handle("/api/v1/integrations/list") is False

    def test_handles_without_version(self, handler):
        assert handler.can_handle("/api/integrations/health") is True


class TestGetHealth:
    """Tests for GET /api/v1/integrations/health."""

    def test_returns_200(self, handler, mock_handler):
        result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_has_expected_fields(self, handler, mock_handler):
        result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
        body = parse_body(result)
        assert "integrations" in body
        assert "total" in body
        assert "configured" in body
        assert "healthy" in body

    def test_all_integrations_listed(self, handler, mock_handler):
        result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
        body = parse_body(result)
        names = {i["name"] for i in body["integrations"]}
        assert "slack" in names
        assert "email" in names
        assert "discord" in names
        assert "teams" in names
        assert "zapier" in names
        assert body["total"] == 5

    def test_integration_fields_present(self, handler, mock_handler):
        result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
        body = parse_body(result)
        for integration in body["integrations"]:
            assert "name" in integration
            assert "configured" in integration
            assert "module_available" in integration
            assert "healthy" in integration
            assert "last_check" in integration

    def test_unconfigured_by_default(self, handler, mock_handler):
        """Without env vars, integrations should show as not configured."""
        # Ensure no integration env vars are set
        env_vars = [
            "SLACK_WEBHOOK_URL",
            "SLACK_BOT_TOKEN",
            "DISCORD_WEBHOOK_URL",
            "DISCORD_BOT_TOKEN",
            "TEAMS_WEBHOOK_URL",
            "MS_TEAMS_WEBHOOK",
            "ZAPIER_WEBHOOK_URL",
            "ZAPIER_API_KEY",
        ]
        with patch.dict(os.environ, {}, clear=False):
            for var in env_vars:
                os.environ.pop(var, None)

            result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
            body = parse_body(result)

            slack = next(i for i in body["integrations"] if i["name"] == "slack")
            assert slack["configured"] is False
            discord = next(i for i in body["integrations"] if i["name"] == "discord")
            assert discord["configured"] is False

    def test_configured_when_env_var_set(self, mock_handler):
        """Integration shows as configured when env var is present."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            handler = IntegrationHealthHandler(ctx={})
            result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
            body = parse_body(result)
            slack = next(i for i in body["integrations"] if i["name"] == "slack")
            assert slack["configured"] is True

    def test_healthy_with_connector_in_context(self, mock_handler):
        """Integration shows healthy when connector reports healthy."""
        from datetime import datetime

        mock_connector = MagicMock()
        mock_connector.healthy = True
        mock_connector.last_check = datetime(2026, 2, 14, 10, 0, 0)

        handler = IntegrationHealthHandler(ctx={"connectors": {"slack": mock_connector}})
        result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
        body = parse_body(result)
        slack = next(i for i in body["integrations"] if i["name"] == "slack")
        assert slack["healthy"] is True
        assert slack["last_check"] is not None

    def test_unhealthy_without_connector(self, handler, mock_handler):
        """Integration shows unhealthy when no connector present."""
        result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
        body = parse_body(result)
        for integration in body["integrations"]:
            assert integration["healthy"] is False

    def test_configured_count(self, mock_handler):
        """configured count reflects number of configured integrations."""
        with patch.dict(
            os.environ,
            {
                "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "DISCORD_BOT_TOKEN": "test-token",
            },
        ):
            handler = IntegrationHealthHandler(ctx={})
            result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
            body = parse_body(result)
            assert body["configured"] >= 2

    def test_healthy_count(self, mock_handler):
        """healthy count reflects number of healthy integrations."""
        mock_conn = MagicMock()
        mock_conn.healthy = True
        mock_conn.last_check = None

        handler = IntegrationHealthHandler(
            ctx={"connectors": {"slack": mock_conn, "email": mock_conn}}
        )
        result = handler.handle("/api/v1/integrations/health", {}, mock_handler)
        body = parse_body(result)
        assert body["healthy"] == 2


class TestUnhandledRoutes:
    """Tests for paths not handled by this handler."""

    def test_unrelated_path_returns_none(self, handler, mock_handler):
        result = handler.handle("/api/v1/admin/health", {}, mock_handler)
        assert result is None
