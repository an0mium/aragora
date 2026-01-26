"""
Tests for external automation integrations (Zapier, Make, n8n).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.integrations.zapier import (
    ZapierIntegration,
    ZapierApp,
    ZapierTrigger,
    get_zapier_integration,
)
from aragora.integrations.make import (
    MakeIntegration,
    MakeConnection,
    MakeWebhook,
    get_make_integration,
)
from aragora.integrations.n8n import (
    N8nIntegration,
    N8nCredential,
    N8nWebhook,
    N8nResourceType,
    N8nOperation,
    get_n8n_integration,
)


# =============================================================================
# Zapier Integration Tests
# =============================================================================


class TestZapierIntegration:
    """Tests for Zapier integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.zapier = ZapierIntegration()

    def test_create_app(self):
        """Test creating a Zapier app."""
        app = self.zapier.create_app("workspace-123")

        assert app is not None
        assert app.workspace_id == "workspace-123"
        assert app.api_key.startswith("zap_")
        assert app.api_secret is not None
        assert app.active is True
        assert len(app.triggers) == 0

    def test_get_app(self):
        """Test getting a Zapier app by ID."""
        app = self.zapier.create_app("workspace-123")
        retrieved = self.zapier.get_app(app.id)

        assert retrieved is not None
        assert retrieved.id == app.id

    def test_get_app_by_key(self):
        """Test getting a Zapier app by API key."""
        app = self.zapier.create_app("workspace-123")
        retrieved = self.zapier.get_app_by_key(app.api_key)

        assert retrieved is not None
        assert retrieved.id == app.id

    def test_list_apps(self):
        """Test listing Zapier apps."""
        self.zapier.create_app("workspace-1")
        self.zapier.create_app("workspace-2")
        self.zapier.create_app("workspace-1")

        all_apps = self.zapier.list_apps()
        assert len(all_apps) == 3

        ws1_apps = self.zapier.list_apps("workspace-1")
        assert len(ws1_apps) == 2

    def test_delete_app(self):
        """Test deleting a Zapier app."""
        app = self.zapier.create_app("workspace-123")
        assert self.zapier.get_app(app.id) is not None

        result = self.zapier.delete_app(app.id)
        assert result is True
        assert self.zapier.get_app(app.id) is None

    def test_subscribe_trigger(self):
        """Test subscribing to a Zapier trigger."""
        app = self.zapier.create_app("workspace-123")

        trigger = self.zapier.subscribe_trigger(
            app_id=app.id,
            trigger_type="debate_completed",
            webhook_url="https://hooks.zapier.com/test",
            workspace_id="workspace-123",
            min_confidence=0.8,
        )

        assert trigger is not None
        assert trigger.trigger_type == "debate_completed"
        assert trigger.webhook_url == "https://hooks.zapier.com/test"
        assert trigger.min_confidence == 0.8

    def test_subscribe_trigger_invalid_type(self):
        """Test subscribing to an invalid trigger type."""
        app = self.zapier.create_app("workspace-123")

        trigger = self.zapier.subscribe_trigger(
            app_id=app.id,
            trigger_type="invalid_trigger",
            webhook_url="https://hooks.zapier.com/test",
        )

        assert trigger is None

    def test_unsubscribe_trigger(self):
        """Test unsubscribing from a Zapier trigger."""
        app = self.zapier.create_app("workspace-123")
        trigger = self.zapier.subscribe_trigger(
            app_id=app.id,
            trigger_type="debate_completed",
            webhook_url="https://hooks.zapier.com/test",
        )

        result = self.zapier.unsubscribe_trigger(app.id, trigger.id)
        assert result is True
        assert len(self.zapier.list_triggers(app.id)) == 0

    def test_trigger_matches_event(self):
        """Test trigger event matching."""
        trigger = ZapierTrigger(
            id="trigger-1",
            trigger_type="debate_completed",
            webhook_url="https://test.com",
            api_key="test-key",
            workspace_id="workspace-123",
            min_confidence=0.8,
        )

        # Matching event
        assert (
            trigger.matches_event(
                {
                    "workspace_id": "workspace-123",
                    "confidence": 0.9,
                }
            )
            is True
        )

        # Wrong workspace
        assert (
            trigger.matches_event(
                {
                    "workspace_id": "other-workspace",
                    "confidence": 0.9,
                }
            )
            is False
        )

        # Low confidence
        assert (
            trigger.matches_event(
                {
                    "workspace_id": "workspace-123",
                    "confidence": 0.5,
                }
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_fire_trigger(self):
        """Test firing Zapier triggers."""
        app = self.zapier.create_app("workspace-123")
        self.zapier.subscribe_trigger(
            app_id=app.id,
            trigger_type="debate_completed",
            webhook_url="https://hooks.zapier.com/test",
        )

        with patch.object(self.zapier, "_fire_single_trigger", new_callable=AsyncMock) as mock_fire:
            mock_fire.return_value = True

            count = await self.zapier.fire_trigger(
                "debate_completed",
                {"id": "event-1", "workspace_id": "workspace-123"},
            )

            assert count == 1
            mock_fire.assert_called_once()

    def test_authenticate_request(self):
        """Test request authentication."""
        app = self.zapier.create_app("workspace-123")

        authenticated = self.zapier.authenticate_request(app.api_key)
        assert authenticated is not None
        assert authenticated.id == app.id

        # Invalid key
        invalid = self.zapier.authenticate_request("invalid-key")
        assert invalid is None


# =============================================================================
# Make Integration Tests
# =============================================================================


class TestMakeIntegration:
    """Tests for Make (Integromat) integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.make = MakeIntegration()

    def test_create_connection(self):
        """Test creating a Make connection."""
        conn = self.make.create_connection("workspace-123")

        assert conn is not None
        assert conn.workspace_id == "workspace-123"
        assert conn.api_key.startswith("make_")
        assert conn.active is True
        assert len(conn.webhooks) == 0

    def test_get_connection(self):
        """Test getting a Make connection by ID."""
        conn = self.make.create_connection("workspace-123")
        retrieved = self.make.get_connection(conn.id)

        assert retrieved is not None
        assert retrieved.id == conn.id

    def test_get_connection_by_key(self):
        """Test getting a Make connection by API key."""
        conn = self.make.create_connection("workspace-123")
        retrieved = self.make.get_connection_by_key(conn.api_key)

        assert retrieved is not None
        assert retrieved.id == conn.id

    def test_list_connections(self):
        """Test listing Make connections."""
        self.make.create_connection("workspace-1")
        self.make.create_connection("workspace-2")

        all_conns = self.make.list_connections()
        assert len(all_conns) == 2

    def test_delete_connection(self):
        """Test deleting a Make connection."""
        conn = self.make.create_connection("workspace-123")
        assert self.make.get_connection(conn.id) is not None

        result = self.make.delete_connection(conn.id)
        assert result is True
        assert self.make.get_connection(conn.id) is None

    def test_register_webhook(self):
        """Test registering a Make webhook."""
        conn = self.make.create_connection("workspace-123")

        webhook = self.make.register_webhook(
            conn_id=conn.id,
            module_type="watch_debates",
            webhook_url="https://hook.make.com/test",
            workspace_id="workspace-123",
        )

        assert webhook is not None
        assert webhook.module_type == "watch_debates"
        assert webhook.webhook_url == "https://hook.make.com/test"

    def test_register_webhook_invalid_module(self):
        """Test registering a webhook with invalid module type."""
        conn = self.make.create_connection("workspace-123")

        webhook = self.make.register_webhook(
            conn_id=conn.id,
            module_type="invalid_module",
            webhook_url="https://hook.make.com/test",
        )

        assert webhook is None

    def test_unregister_webhook(self):
        """Test unregistering a Make webhook."""
        conn = self.make.create_connection("workspace-123")
        webhook = self.make.register_webhook(
            conn_id=conn.id,
            module_type="watch_debates",
            webhook_url="https://hook.make.com/test",
        )

        result = self.make.unregister_webhook(conn.id, webhook.id)
        assert result is True
        assert len(self.make.list_webhooks(conn.id)) == 0

    def test_webhook_matches_event(self):
        """Test webhook event matching."""
        webhook = MakeWebhook(
            id="webhook-1",
            module_type="watch_debates",
            webhook_url="https://test.com",
            workspace_id="workspace-123",
            event_filter={"status": "completed"},
        )

        # Matching event
        assert (
            webhook.matches_event(
                {
                    "workspace_id": "workspace-123",
                    "status": "completed",
                }
            )
            is True
        )

        # Wrong workspace
        assert (
            webhook.matches_event(
                {
                    "workspace_id": "other-workspace",
                    "status": "completed",
                }
            )
            is False
        )

        # Wrong filter value
        assert (
            webhook.matches_event(
                {
                    "workspace_id": "workspace-123",
                    "status": "pending",
                }
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_trigger_webhooks(self):
        """Test triggering Make webhooks."""
        conn = self.make.create_connection("workspace-123")
        self.make.register_webhook(
            conn_id=conn.id,
            module_type="watch_debates",
            webhook_url="https://hook.make.com/test",
        )

        with patch.object(
            self.make, "_trigger_single_webhook", new_callable=AsyncMock
        ) as mock_trigger:
            mock_trigger.return_value = True

            count = await self.make.trigger_webhooks(
                "watch_debates",
                {"id": "event-1"},
            )

            assert count == 1
            mock_trigger.assert_called_once()

    def test_test_connection(self):
        """Test connection testing."""
        conn = self.make.create_connection("workspace-123")

        result = self.make.test_connection(conn.id)
        assert result["success"] is True
        assert result["connection_id"] == conn.id

        # Non-existent connection
        result = self.make.test_connection("non-existent")
        assert result["success"] is False

    def test_module_types(self):
        """Test that module types are properly defined."""
        assert "watch_debates" in self.make.MODULE_TYPES
        assert "create_debate" in self.make.MODULE_TYPES

        # Check module metadata
        watch = self.make.MODULE_TYPES["watch_debates"]
        assert watch["type"] == "trigger"
        assert watch["instant"] is True


# =============================================================================
# n8n Integration Tests
# =============================================================================


class TestN8nIntegration:
    """Tests for n8n integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.n8n = N8nIntegration()

    def test_create_credential(self):
        """Test creating an n8n credential."""
        cred = self.n8n.create_credential("workspace-123")

        assert cred is not None
        assert cred.workspace_id == "workspace-123"
        assert cred.api_key.startswith("n8n_")
        assert cred.active is True
        assert len(cred.webhooks) == 0

    def test_get_credential(self):
        """Test getting an n8n credential by ID."""
        cred = self.n8n.create_credential("workspace-123")
        retrieved = self.n8n.get_credential(cred.id)

        assert retrieved is not None
        assert retrieved.id == cred.id

    def test_get_credential_by_key(self):
        """Test getting an n8n credential by API key."""
        cred = self.n8n.create_credential("workspace-123")
        retrieved = self.n8n.get_credential_by_key(cred.api_key)

        assert retrieved is not None
        assert retrieved.id == cred.id

    def test_list_credentials(self):
        """Test listing n8n credentials."""
        self.n8n.create_credential("workspace-1")
        self.n8n.create_credential("workspace-2")

        all_creds = self.n8n.list_credentials()
        assert len(all_creds) == 2

    def test_delete_credential(self):
        """Test deleting an n8n credential."""
        cred = self.n8n.create_credential("workspace-123")
        assert self.n8n.get_credential(cred.id) is not None

        result = self.n8n.delete_credential(cred.id)
        assert result is True
        assert self.n8n.get_credential(cred.id) is None

    def test_register_webhook(self):
        """Test registering an n8n webhook."""
        cred = self.n8n.create_credential("workspace-123")

        webhook = self.n8n.register_webhook(
            cred_id=cred.id,
            events=["debate_end", "consensus"],
            workflow_id="workflow-1",
            workspace_id="workspace-123",
        )

        assert webhook is not None
        assert "debate_end" in webhook.events
        assert webhook.webhook_path.startswith("/n8n/webhook/")

    def test_register_webhook_invalid_events(self):
        """Test registering a webhook with invalid events."""
        cred = self.n8n.create_credential("workspace-123")

        webhook = self.n8n.register_webhook(
            cred_id=cred.id,
            events=["invalid_event"],
        )

        assert webhook is None

    def test_unregister_webhook(self):
        """Test unregistering an n8n webhook."""
        cred = self.n8n.create_credential("workspace-123")
        webhook = self.n8n.register_webhook(
            cred_id=cred.id,
            events=["debate_end"],
        )

        result = self.n8n.unregister_webhook(cred.id, webhook.id)
        assert result is True
        assert len(self.n8n.list_webhooks(cred.id)) == 0

    def test_get_webhook_by_path(self):
        """Test getting a webhook by path."""
        cred = self.n8n.create_credential("workspace-123")
        webhook = self.n8n.register_webhook(
            cred_id=cred.id,
            events=["debate_end"],
        )

        retrieved = self.n8n.get_webhook_by_path(webhook.webhook_path)
        assert retrieved is not None
        assert retrieved.id == webhook.id

    def test_webhook_matches_event(self):
        """Test webhook event matching."""
        webhook = N8nWebhook(
            id="webhook-1",
            webhook_path="/n8n/webhook/test",
            events=["debate_end", "consensus"],
            workspace_id="workspace-123",
        )

        # Matching event
        assert (
            webhook.matches_event(
                "debate_end",
                {
                    "workspace_id": "workspace-123",
                },
            )
            is True
        )

        # Wrong workspace
        assert (
            webhook.matches_event(
                "debate_end",
                {
                    "workspace_id": "other-workspace",
                },
            )
            is False
        )

        # Event not in list
        assert (
            webhook.matches_event(
                "debate_start",
                {
                    "workspace_id": "workspace-123",
                },
            )
            is False
        )

    def test_webhook_wildcard_events(self):
        """Test webhook with wildcard events."""
        webhook = N8nWebhook(
            id="webhook-1",
            webhook_path="/n8n/webhook/test",
            events=["*"],
        )

        # Should match any event in the event types
        assert webhook.matches_event("debate_end", {}) is True
        assert webhook.matches_event("consensus", {}) is True

    @pytest.mark.asyncio
    async def test_dispatch_event(self):
        """Test dispatching n8n events."""
        cred = self.n8n.create_credential("workspace-123")
        self.n8n.register_webhook(
            cred_id=cred.id,
            events=["debate_end"],
        )

        with patch.object(
            self.n8n, "_dispatch_to_webhook", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = True

            count = await self.n8n.dispatch_event(
                "debate_end",
                {"id": "event-1"},
            )

            assert count == 1
            mock_dispatch.assert_called_once()

    def test_get_node_definition(self):
        """Test getting n8n node definition."""
        node_def = self.n8n.get_node_definition()

        assert node_def["displayName"] == "Aragora"
        assert node_def["name"] == "aragora"
        assert "properties" in node_def

    def test_get_trigger_node_definition(self):
        """Test getting n8n trigger node definition."""
        trigger_def = self.n8n.get_trigger_node_definition()

        assert trigger_def["displayName"] == "Aragora Trigger"
        assert trigger_def["name"] == "aragoraTrigger"
        assert "webhooks" in trigger_def

    def test_get_credential_definition(self):
        """Test getting n8n credential definition."""
        cred_def = self.n8n.get_credential_definition()

        assert cred_def["name"] == "aragoraApi"
        assert "properties" in cred_def

    @pytest.mark.asyncio
    async def test_execute_operation(self):
        """Test executing n8n operations."""
        cred = self.n8n.create_credential("workspace-123")

        result = await self.n8n.execute_operation(
            cred_id=cred.id,
            resource=N8nResourceType.DEBATE,
            operation=N8nOperation.CREATE,
            parameters={"question": "Test question"},
        )

        assert result["success"] is True
        assert result["resource"] == "debate"
        assert result["operation"] == "create"

    def test_event_types(self):
        """Test that event types are properly defined."""
        assert "debate_end" in self.n8n.EVENT_TYPES
        assert "consensus" in self.n8n.EVENT_TYPES
        assert "gauntlet_complete" in self.n8n.EVENT_TYPES


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletons:
    """Test singleton factory functions."""

    def test_get_zapier_integration(self):
        """Test Zapier integration singleton."""
        # Reset global singleton
        import aragora.integrations.zapier as zapier_mod

        zapier_mod._zapier_integration = None

        zapier1 = get_zapier_integration()
        zapier2 = get_zapier_integration()

        assert zapier1 is zapier2

    def test_get_make_integration(self):
        """Test Make integration singleton."""
        # Reset global singleton
        import aragora.integrations.make as make_mod

        make_mod._make_integration = None

        make1 = get_make_integration()
        make2 = get_make_integration()

        assert make1 is make2

    def test_get_n8n_integration(self):
        """Test n8n integration singleton."""
        # Reset global singleton
        import aragora.integrations.n8n as n8n_mod

        n8n_mod._n8n_integration = None

        n8n1 = get_n8n_integration()
        n8n2 = get_n8n_integration()

        assert n8n1 is n8n2
