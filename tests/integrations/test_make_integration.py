"""Tests for Make (Integromat) integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.make import (
    MakeConnection,
    MakeIntegration,
    MakeWebhook,
    get_make_integration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration():
    return MakeIntegration(api_base="https://test.aragora.ai")


@pytest.fixture
def integration_with_connection(integration):
    conn = integration.create_connection("workspace-1")
    return integration, conn


# =============================================================================
# MakeWebhook Tests
# =============================================================================


class TestMakeWebhook:
    def test_matches_event_no_filters(self):
        webhook = MakeWebhook(
            id="wh1", module_type="watch_debates", webhook_url="https://hook.make.com/test"
        )
        assert webhook.matches_event({"type": "debate_start"}) is True

    def test_matches_event_workspace_filter_match(self):
        webhook = MakeWebhook(
            id="wh1",
            module_type="watch_debates",
            webhook_url="https://test.webhook.com/hook",
            workspace_id="ws-1",
        )
        assert webhook.matches_event({"workspace_id": "ws-1"}) is True

    def test_matches_event_workspace_filter_mismatch(self):
        webhook = MakeWebhook(
            id="wh1",
            module_type="watch_debates",
            webhook_url="https://test.webhook.com/hook",
            workspace_id="ws-1",
        )
        assert webhook.matches_event({"workspace_id": "ws-2"}) is False

    def test_matches_event_filter(self):
        webhook = MakeWebhook(
            id="wh1",
            module_type="watch_debates",
            webhook_url="https://test.webhook.com/hook",
            event_filter={"status": "completed"},
        )
        assert webhook.matches_event({"status": "completed"}) is True
        assert webhook.matches_event({"status": "started"}) is False


# =============================================================================
# MakeConnection Tests
# =============================================================================


class TestMakeConnection:
    def test_defaults(self):
        conn = MakeConnection(id="c1", workspace_id="ws1", api_key="key1")
        assert conn.active is True
        assert conn.total_operations == 0
        assert conn.webhooks == {}


# =============================================================================
# MakeIntegration Tests
# =============================================================================


class TestMakeIntegration:
    def test_initialization(self, integration):
        assert integration.api_base == "https://test.aragora.ai"
        assert integration.is_configured is False

    def test_is_configured_with_connection(self, integration_with_connection):
        integ, _ = integration_with_connection
        assert integ.is_configured is True

    # --- Connection Management ---

    def test_create_connection(self, integration):
        conn = integration.create_connection("ws-1")
        assert conn.workspace_id == "ws-1"
        assert conn.api_key.startswith("make_")
        assert conn.active is True

    def test_get_connection(self, integration_with_connection):
        integ, conn = integration_with_connection
        retrieved = integ.get_connection(conn.id)
        assert retrieved is not None
        assert retrieved.id == conn.id

    def test_get_connection_not_found(self, integration):
        assert integration.get_connection("nonexistent") is None

    def test_get_connection_by_key(self, integration_with_connection):
        integ, conn = integration_with_connection
        retrieved = integ.get_connection_by_key(conn.api_key)
        assert retrieved is not None
        assert retrieved.id == conn.id

    def test_get_connection_by_key_not_found(self, integration):
        assert integration.get_connection_by_key("nonexistent") is None

    def test_list_connections(self, integration):
        integration.create_connection("ws-1")
        integration.create_connection("ws-1")
        integration.create_connection("ws-2")
        all_conns = integration.list_connections()
        assert len(all_conns) == 3
        ws1_conns = integration.list_connections(workspace_id="ws-1")
        assert len(ws1_conns) == 2

    def test_delete_connection(self, integration_with_connection):
        integ, conn = integration_with_connection
        assert integ.delete_connection(conn.id) is True
        assert integ.get_connection(conn.id) is None

    def test_delete_nonexistent_connection(self, integration):
        assert integration.delete_connection("nonexistent") is False

    # --- Webhook Management ---

    def test_register_webhook(self, integration_with_connection):
        integ, conn = integration_with_connection
        webhook = integ.register_webhook(conn.id, "watch_debates", "https://hook.make.com/test")
        assert webhook is not None
        assert webhook.module_type == "watch_debates"

    def test_register_webhook_invalid_connection(self, integration):
        result = integration.register_webhook(
            "bad", "watch_debates", "https://test.webhook.com/hook"
        )
        assert result is None

    def test_register_webhook_invalid_module(self, integration_with_connection):
        integ, conn = integration_with_connection
        result = integ.register_webhook(
            conn.id, "nonexistent_module", "https://test.webhook.com/hook"
        )
        assert result is None

    def test_register_webhook_non_trigger(self, integration_with_connection):
        integ, conn = integration_with_connection
        result = integ.register_webhook(conn.id, "create_debate", "https://test.webhook.com/hook")
        assert result is None

    def test_unregister_webhook(self, integration_with_connection):
        integ, conn = integration_with_connection
        webhook = integ.register_webhook(conn.id, "watch_debates", "https://test.webhook.com/hook")
        assert integ.unregister_webhook(conn.id, webhook.id) is True

    def test_unregister_webhook_not_found(self, integration_with_connection):
        integ, conn = integration_with_connection
        assert integ.unregister_webhook(conn.id, "nonexistent") is False

    def test_unregister_webhook_bad_connection(self, integration):
        assert integration.unregister_webhook("bad", "wh1") is False

    def test_list_webhooks(self, integration_with_connection):
        integ, conn = integration_with_connection
        integ.register_webhook(conn.id, "watch_debates", "https://test.webhook.com/hook1")
        integ.register_webhook(conn.id, "watch_consensus", "https://test.webhook.com/hook2")
        webhooks = integ.list_webhooks(conn.id)
        assert len(webhooks) == 2

    def test_list_webhooks_not_found(self, integration):
        assert integration.list_webhooks("bad") == []

    # --- Action Execution ---

    @pytest.mark.asyncio
    async def test_execute_action(self, integration_with_connection):
        integ, conn = integration_with_connection
        result = await integ.execute_action(conn.id, "create_debate", {"topic": "Test"})
        assert result["success"] is True
        assert result["action"] == "create_debate"

    @pytest.mark.asyncio
    async def test_execute_action_no_connection(self, integration):
        result = await integration.execute_action("bad", "create_debate", {})
        assert result["error"] == "Connection not found"

    @pytest.mark.asyncio
    async def test_execute_action_invalid_type(self, integration_with_connection):
        integ, conn = integration_with_connection
        result = await integ.execute_action(conn.id, "nonexistent", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_action_non_action(self, integration_with_connection):
        integ, conn = integration_with_connection
        result = await integ.execute_action(conn.id, "watch_debates", {})
        assert "error" in result

    # --- Authentication ---

    def test_authenticate_request(self, integration_with_connection):
        integ, conn = integration_with_connection
        result = integ.authenticate_request(conn.api_key)
        assert result is not None
        assert result.id == conn.id

    def test_authenticate_request_invalid(self, integration):
        assert integration.authenticate_request("bad_key") is None

    # --- Connection Testing ---

    def test_test_connection(self, integration_with_connection):
        integ, conn = integration_with_connection
        result = integ.test_connection(conn.id)
        assert result["success"] is True
        assert result["connection_id"] == conn.id

    def test_test_connection_not_found(self, integration):
        result = integration.test_connection("bad")
        assert result["success"] is False

    # --- Trigger Dispatch ---

    @pytest.mark.asyncio
    async def test_trigger_webhooks(self, integration_with_connection):
        integ, conn = integration_with_connection
        integ.register_webhook(conn.id, "watch_debates", "https://hook.make.com/test")

        with patch.object(
            integ, "_trigger_single_webhook", new_callable=AsyncMock, return_value=True
        ):
            count = await integ.trigger_webhooks("watch_debates", {"type": "debate_start"})
            assert count == 1

    @pytest.mark.asyncio
    async def test_trigger_webhooks_no_match(self, integration_with_connection):
        integ, conn = integration_with_connection
        integ.register_webhook(conn.id, "watch_consensus", "https://test.webhook.com/hook")
        count = await integ.trigger_webhooks("watch_debates", {})
        assert count == 0

    @pytest.mark.asyncio
    async def test_send_message_no_url(self, integration):
        result = await integration.send_message("test")
        assert result is False

    # --- Module Definitions ---

    def test_module_types(self, integration):
        assert "watch_debates" in MakeIntegration.MODULE_TYPES
        assert "create_debate" in MakeIntegration.MODULE_TYPES
        assert MakeIntegration.MODULE_TYPES["watch_debates"]["type"] == "trigger"
        assert MakeIntegration.MODULE_TYPES["create_debate"]["type"] == "action"


# =============================================================================
# Singleton Tests
# =============================================================================


class TestGetMakeIntegration:
    def test_singleton(self):
        # Reset global
        import aragora.integrations.make as mod

        mod._make_integration = None
        integ1 = get_make_integration()
        integ2 = get_make_integration()
        assert integ1 is integ2
        mod._make_integration = None  # cleanup
