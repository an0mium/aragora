"""Tests for MCP integrations tools execution logic."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.integrations import (
    get_integration_events_tool,
    list_integrations_tool,
    test_integration_tool as _test_integration_tool,
    trigger_external_webhook_tool,
)


def _make_mock_module(func_name, return_value):
    """Create a mock module with a single factory function returning the given value."""
    mod = MagicMock()
    getattr(mod, func_name).return_value = return_value
    return mod


# Keys for the three integration modules in sys.modules
_ZAPIER_MOD = "aragora.integrations.zapier"
_MAKE_MOD = "aragora.integrations.make"
_N8N_MOD = "aragora.integrations.n8n"


class TestTriggerExternalWebhookTool:
    """Tests for trigger_external_webhook_tool."""

    @pytest.mark.asyncio
    async def test_invalid_platform(self):
        """Test trigger with invalid platform."""
        result = await trigger_external_webhook_tool(platform="invalid", event_type="test")
        assert "error" in result
        assert "Invalid platform" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_json_data(self):
        """Test trigger with invalid JSON data."""
        result = await trigger_external_webhook_tool(
            platform="zapier", event_type="test", data="not json"
        )
        assert "error" in result
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_trigger_zapier_success(self):
        """Test successful Zapier webhook trigger."""
        mock_zapier = AsyncMock()
        mock_zapier.fire_trigger.return_value = 2

        mock_module = _make_mock_module("get_zapier_integration", mock_zapier)

        with patch.dict(sys.modules, {_ZAPIER_MOD: mock_module}):
            result = await trigger_external_webhook_tool(
                platform="zapier",
                event_type="debate_completed",
                data='{"debate_id": "d-001"}',
            )

        assert result["platform"] == "zapier"
        assert result["triggered"] is True
        assert result["webhooks_triggered"] == 2

    @pytest.mark.asyncio
    async def test_trigger_make_success(self):
        """Test successful Make webhook trigger."""
        mock_make = AsyncMock()
        mock_make.trigger_webhooks.return_value = 1

        mock_module = _make_mock_module("get_make_integration", mock_make)

        with patch.dict(sys.modules, {_MAKE_MOD: mock_module}):
            result = await trigger_external_webhook_tool(
                platform="make",
                event_type="audit_completed",
            )

        assert result["platform"] == "make"
        assert result["triggered"] is True

    @pytest.mark.asyncio
    async def test_trigger_n8n_success(self):
        """Test successful n8n webhook trigger."""
        mock_n8n = AsyncMock()
        mock_n8n.dispatch_event.return_value = 0

        mock_module = _make_mock_module("get_n8n_integration", mock_n8n)

        with patch.dict(sys.modules, {_N8N_MOD: mock_module}):
            result = await trigger_external_webhook_tool(
                platform="n8n",
                event_type="consensus_reached",
            )

        assert result["platform"] == "n8n"
        assert result["triggered"] is False
        assert result["webhooks_triggered"] == 0

    @pytest.mark.asyncio
    async def test_trigger_import_error(self):
        """Test trigger when integration module unavailable."""
        # Remove the module from sys.modules so the import fails
        with patch.dict(sys.modules, {_ZAPIER_MOD: None}):
            result = await trigger_external_webhook_tool(platform="zapier", event_type="test")

        assert "error" in result
        assert "not available" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_trigger_adds_metadata(self):
        """Test that trigger adds timestamp and event_type to data."""
        mock_zapier = AsyncMock()
        mock_zapier.fire_trigger.return_value = 1

        captured_data = {}

        async def capture_fire_trigger(event_type, data):
            captured_data.update(data)
            return 1

        mock_zapier.fire_trigger = capture_fire_trigger

        mock_module = _make_mock_module("get_zapier_integration", mock_zapier)

        with patch.dict(sys.modules, {_ZAPIER_MOD: mock_module}):
            await trigger_external_webhook_tool(
                platform="zapier",
                event_type="test_event",
                data='{"key": "value"}',
            )

        assert "timestamp" in captured_data
        assert captured_data["event_type"] == "test_event"
        assert captured_data["source"] == "mcp_tool"
        assert captured_data["key"] == "value"


class TestListIntegrationsTool:
    """Tests for list_integrations_tool."""

    @pytest.mark.asyncio
    async def test_list_all_empty(self):
        """Test list all platforms when none available."""
        # Setting modules to None makes 'from X import Y' raise ImportError
        with patch.dict(
            sys.modules,
            {_ZAPIER_MOD: None, _MAKE_MOD: None, _N8N_MOD: None},
        ):
            result = await list_integrations_tool()

        assert result["total"] == 0
        assert "integrations" in result

    @pytest.mark.asyncio
    async def test_list_zapier_only(self):
        """Test list with zapier filter."""
        mock_zapier = MagicMock()
        mock_app = MagicMock()
        mock_app.id = "app-001"
        mock_app.workspace_id = "ws-001"
        mock_app.active = True
        mock_app.triggers = ["t1", "t2"]
        mock_app.created_at = "2025-01-01"
        mock_zapier.list_apps.return_value = [mock_app]

        mock_module = _make_mock_module("get_zapier_integration", mock_zapier)

        with patch.dict(sys.modules, {_ZAPIER_MOD: mock_module}):
            result = await list_integrations_tool(platform="zapier")

        assert result["total"] == 1
        assert result["platform_filter"] == "zapier"

    @pytest.mark.asyncio
    async def test_list_with_workspace_filter(self):
        """Test list with workspace_id filter."""
        with patch.dict(
            sys.modules,
            {_ZAPIER_MOD: None, _MAKE_MOD: None, _N8N_MOD: None},
        ):
            result = await list_integrations_tool(workspace_id="ws-001")

        assert result["workspace_filter"] == "ws-001"


class TestTestIntegrationTool:
    """Tests for test_integration_tool."""

    @pytest.mark.asyncio
    async def test_invalid_platform(self):
        """Test with invalid platform."""
        result = await _test_integration_tool(platform="invalid", integration_id="123")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_zapier_success(self):
        """Test successful Zapier integration test."""
        mock_zapier = MagicMock()
        mock_app = MagicMock()
        mock_app.active = True
        mock_app.triggers = ["t1"]
        mock_app.trigger_count = 42
        mock_zapier.get_app.return_value = mock_app

        mock_module = _make_mock_module("get_zapier_integration", mock_zapier)

        with patch.dict(sys.modules, {_ZAPIER_MOD: mock_module}):
            result = await _test_integration_tool(platform="zapier", integration_id="app-001")

        assert result["platform"] == "zapier"
        assert result["status"] == "ok"
        assert result["total_triggers_fired"] == 42

    @pytest.mark.asyncio
    async def test_zapier_not_found(self):
        """Test Zapier app not found."""
        mock_zapier = MagicMock()
        mock_zapier.get_app.return_value = None

        mock_module = _make_mock_module("get_zapier_integration", mock_zapier)

        with patch.dict(sys.modules, {_ZAPIER_MOD: mock_module}):
            result = await _test_integration_tool(platform="zapier", integration_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_n8n_inactive(self):
        """Test n8n credential that is inactive."""
        mock_n8n = MagicMock()
        mock_cred = MagicMock()
        mock_cred.active = False
        mock_cred.webhooks = []
        mock_cred.operation_count = 0
        mock_n8n.get_credential.return_value = mock_cred

        mock_module = _make_mock_module("get_n8n_integration", mock_n8n)

        with patch.dict(sys.modules, {_N8N_MOD: mock_module}):
            result = await _test_integration_tool(platform="n8n", integration_id="cred-001")

        assert result["platform"] == "n8n"
        assert result["status"] == "inactive"

    @pytest.mark.asyncio
    async def test_import_error(self):
        """Test integration module not available."""
        with patch.dict(sys.modules, {_MAKE_MOD: None}):
            result = await _test_integration_tool(platform="make", integration_id="conn-001")

        assert "error" in result
        assert "not available" in result["error"].lower()


class TestGetIntegrationEventsTool:
    """Tests for get_integration_events_tool."""

    @pytest.mark.asyncio
    async def test_invalid_platform(self):
        """Test with invalid platform."""
        result = await get_integration_events_tool(platform="invalid")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_zapier_events(self):
        """Test getting Zapier event types."""
        mock_zapier = MagicMock()
        mock_zapier.TRIGGER_TYPES = ["debate_completed", "consensus_reached"]
        mock_zapier.ACTION_TYPES = ["send_notification", "create_report"]

        mock_module = _make_mock_module("get_zapier_integration", mock_zapier)

        with patch.dict(sys.modules, {_ZAPIER_MOD: mock_module}):
            result = await get_integration_events_tool(platform="zapier")

        assert result["platform"] == "zapier"
        assert "debate_completed" in result["trigger_types"]
        assert "send_notification" in result["action_types"]

    @pytest.mark.asyncio
    async def test_n8n_events(self):
        """Test getting n8n event types."""
        mock_n8n = MagicMock()
        mock_n8n.EVENT_TYPES = ["webhook", "schedule", "manual"]

        mock_module = _make_mock_module("get_n8n_integration", mock_n8n)

        with patch.dict(sys.modules, {_N8N_MOD: mock_module}):
            result = await get_integration_events_tool(platform="n8n")

        assert result["platform"] == "n8n"
        assert "webhook" in result["event_types"]

    @pytest.mark.asyncio
    async def test_events_import_error(self):
        """Test events when module unavailable."""
        with patch.dict(sys.modules, {_ZAPIER_MOD: None}):
            result = await get_integration_events_tool(platform="zapier")

        assert "error" in result
