"""Tests for Zapier integration."""

from __future__ import annotations

import hashlib
import hmac
from unittest.mock import AsyncMock, patch

import pytest

from aragora.integrations.zapier import (
    ZapierApp,
    ZapierIntegration,
    ZapierTrigger,
    get_zapier_integration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration():
    return ZapierIntegration(api_base="https://test.aragora.ai")


@pytest.fixture
def integration_with_app(integration):
    app = integration.create_app("workspace-1")
    return integration, app


# =============================================================================
# ZapierTrigger Tests
# =============================================================================


class TestZapierTrigger:
    def test_matches_event_no_filters(self):
        trigger = ZapierTrigger(
            id="t1", trigger_type="debate_completed", webhook_url="url", api_key="key"
        )
        assert trigger.matches_event({"type": "debate"}) is True

    def test_matches_event_workspace_filter(self):
        trigger = ZapierTrigger(
            id="t1",
            trigger_type="debate_completed",
            webhook_url="url",
            api_key="key",
            workspace_id="ws-1",
        )
        assert trigger.matches_event({"workspace_id": "ws-1"}) is True
        assert trigger.matches_event({"workspace_id": "ws-2"}) is False

    def test_matches_event_tag_filter(self):
        trigger = ZapierTrigger(
            id="t1",
            trigger_type="debate_completed",
            webhook_url="url",
            api_key="key",
            debate_tags=["urgent", "security"],
        )
        assert trigger.matches_event({"tags": ["urgent", "other"]}) is True
        assert trigger.matches_event({"tags": ["other"]}) is False
        assert trigger.matches_event({}) is False

    def test_matches_event_confidence_filter(self):
        trigger = ZapierTrigger(
            id="t1",
            trigger_type="consensus_reached",
            webhook_url="url",
            api_key="key",
            min_confidence=0.8,
        )
        assert trigger.matches_event({"confidence": 0.9}) is True
        assert trigger.matches_event({"confidence": 0.7}) is False
        assert trigger.matches_event({}) is False


# =============================================================================
# ZapierApp Tests
# =============================================================================


class TestZapierApp:
    def test_defaults(self):
        app = ZapierApp(id="a1", workspace_id="ws1", api_key="key", api_secret="secret")
        assert app.active is True
        assert app.action_count == 0
        assert app.trigger_count == 0
        assert app.triggers == {}


# =============================================================================
# ZapierIntegration Tests
# =============================================================================


class TestZapierIntegration:
    def test_initialization(self, integration):
        assert integration.api_base == "https://test.aragora.ai"
        assert integration.is_configured is False

    def test_is_configured_with_app(self, integration_with_app):
        integ, _ = integration_with_app
        assert integ.is_configured is True

    # --- App Management ---

    def test_create_app(self, integration):
        app = integration.create_app("ws-1")
        assert app.workspace_id == "ws-1"
        assert app.api_key.startswith("zap_")
        assert app.active is True

    def test_get_app(self, integration_with_app):
        integ, app = integration_with_app
        result = integ.get_app(app.id)
        assert result is not None
        assert result.id == app.id

    def test_get_app_not_found(self, integration):
        assert integration.get_app("bad") is None

    def test_get_app_by_key(self, integration_with_app):
        integ, app = integration_with_app
        result = integ.get_app_by_key(app.api_key)
        assert result is not None

    def test_get_app_by_key_not_found(self, integration):
        assert integration.get_app_by_key("bad") is None

    def test_list_apps(self, integration):
        integration.create_app("ws-1")
        integration.create_app("ws-1")
        integration.create_app("ws-2")
        assert len(integration.list_apps()) == 3
        assert len(integration.list_apps(workspace_id="ws-1")) == 2

    def test_delete_app(self, integration_with_app):
        integ, app = integration_with_app
        assert integ.delete_app(app.id) is True
        assert integ.get_app(app.id) is None

    def test_delete_app_not_found(self, integration):
        assert integration.delete_app("bad") is False

    # --- Trigger Management ---

    def test_subscribe_trigger(self, integration_with_app):
        integ, app = integration_with_app
        trigger = integ.subscribe_trigger(
            app.id, "debate_completed", "https://hooks.zapier.com/test"
        )
        assert trigger is not None
        assert trigger.trigger_type == "debate_completed"

    def test_subscribe_trigger_with_filters(self, integration_with_app):
        integ, app = integration_with_app
        trigger = integ.subscribe_trigger(
            app.id,
            "consensus_reached",
            "url",
            workspace_id="ws-1",
            debate_tags=["urgent"],
            min_confidence=0.8,
        )
        assert trigger.workspace_id == "ws-1"
        assert trigger.debate_tags == ["urgent"]
        assert trigger.min_confidence == 0.8

    def test_subscribe_trigger_invalid_app(self, integration):
        result = integration.subscribe_trigger("bad", "debate_completed", "url")
        assert result is None

    def test_subscribe_trigger_invalid_type(self, integration_with_app):
        integ, app = integration_with_app
        result = integ.subscribe_trigger(app.id, "nonexistent_trigger", "url")
        assert result is None

    def test_unsubscribe_trigger(self, integration_with_app):
        integ, app = integration_with_app
        trigger = integ.subscribe_trigger(app.id, "debate_completed", "url")
        assert integ.unsubscribe_trigger(app.id, trigger.id) is True

    def test_unsubscribe_trigger_not_found(self, integration_with_app):
        integ, app = integration_with_app
        assert integ.unsubscribe_trigger(app.id, "bad") is False

    def test_unsubscribe_trigger_bad_app(self, integration):
        assert integration.unsubscribe_trigger("bad", "t1") is False

    def test_list_triggers(self, integration_with_app):
        integ, app = integration_with_app
        integ.subscribe_trigger(app.id, "debate_completed", "url1")
        integ.subscribe_trigger(app.id, "consensus_reached", "url2")
        assert len(integ.list_triggers(app.id)) == 2

    def test_list_triggers_bad_app(self, integration):
        assert integration.list_triggers("bad") == []

    # --- Trigger Dispatch ---

    @pytest.mark.asyncio
    async def test_fire_trigger(self, integration_with_app):
        integ, app = integration_with_app
        integ.subscribe_trigger(app.id, "debate_completed", "https://hooks.zapier.com/test")

        with patch.object(integ, "_fire_single_trigger", new_callable=AsyncMock, return_value=True):
            count = await integ.fire_trigger("debate_completed", {"id": "e1"})
            assert count == 1

    @pytest.mark.asyncio
    async def test_fire_trigger_no_match(self, integration_with_app):
        integ, app = integration_with_app
        integ.subscribe_trigger(app.id, "consensus_reached", "url")
        count = await integ.fire_trigger("debate_completed", {})
        assert count == 0

    @pytest.mark.asyncio
    async def test_fire_trigger_inactive_app(self, integration_with_app):
        integ, app = integration_with_app
        app.active = False
        integ.subscribe_trigger(app.id, "debate_completed", "url")
        count = await integ.fire_trigger("debate_completed", {})
        assert count == 0

    def test_format_trigger_payload(self, integration_with_app):
        integ, app = integration_with_app
        trigger = ZapierTrigger(
            id="t1", trigger_type="debate_completed", webhook_url="url", api_key="key"
        )
        payload = integ._format_trigger_payload(trigger, {"id": "e1", "timestamp": 12345})
        # Zapier expects a list
        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["trigger_type"] == "debate_completed"
        assert payload[0]["id"] == "e1"

    # --- Polling ---

    def test_get_polling_data(self, integration_with_app):
        integ, app = integration_with_app
        data = integ.get_polling_data(app.id, "debate_completed")
        assert data == []

    # --- Authentication ---

    def test_authenticate_request(self, integration_with_app):
        integ, app = integration_with_app
        result = integ.authenticate_request(app.api_key)
        assert result is not None

    def test_authenticate_request_invalid(self, integration):
        assert integration.authenticate_request("bad") is None

    def test_verify_signature(self, integration):
        payload = b'{"test": true}'
        secret = "test_secret"
        expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        assert integration.verify_signature(payload, expected, secret) is True

    def test_verify_signature_invalid(self, integration):
        assert integration.verify_signature(b"data", "bad", "secret") is False

    @pytest.mark.asyncio
    async def test_send_message_no_url(self, integration):
        result = await integration.send_message("test")
        assert result is False

    # --- Constants ---

    def test_trigger_types(self):
        assert "debate_completed" in ZapierIntegration.TRIGGER_TYPES
        assert "consensus_reached" in ZapierIntegration.TRIGGER_TYPES
        assert "breakpoint_hit" in ZapierIntegration.TRIGGER_TYPES

    def test_action_types(self):
        assert "start_debate" in ZapierIntegration.ACTION_TYPES
        assert "get_debate" in ZapierIntegration.ACTION_TYPES
        assert "trigger_gauntlet" in ZapierIntegration.ACTION_TYPES


# =============================================================================
# Singleton Tests
# =============================================================================


class TestGetZapierIntegration:
    def test_singleton(self):
        import aragora.integrations.zapier as mod

        mod._zapier_integration = None
        integ1 = get_zapier_integration()
        integ2 = get_zapier_integration()
        assert integ1 is integ2
        mod._zapier_integration = None
