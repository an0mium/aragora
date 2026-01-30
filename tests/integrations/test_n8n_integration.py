"""Tests for n8n integration."""

from __future__ import annotations

import hashlib
import hmac
from unittest.mock import AsyncMock, patch

import pytest

from aragora.integrations.n8n import (
    N8N_NODE_DEFINITION,
    N8N_TRIGGER_NODE_DEFINITION,
    N8nCredential,
    N8nIntegration,
    N8nOperation,
    N8nResourceType,
    N8nWebhook,
    get_n8n_integration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration():
    return N8nIntegration(api_base="https://test.aragora.ai")


@pytest.fixture
def integration_with_cred(integration):
    cred = integration.create_credential("workspace-1")
    return integration, cred


# =============================================================================
# N8nResourceType Tests
# =============================================================================


class TestN8nResourceType:
    def test_enum_values(self):
        assert N8nResourceType.DEBATE.value == "debate"
        assert N8nResourceType.AGENT.value == "agent"
        assert N8nResourceType.EVIDENCE.value == "evidence"
        assert N8nResourceType.GAUNTLET.value == "gauntlet"
        assert N8nResourceType.KNOWLEDGE.value == "knowledge"


class TestN8nOperation:
    def test_enum_values(self):
        assert N8nOperation.CREATE.value == "create"
        assert N8nOperation.GET.value == "get"
        assert N8nOperation.GET_ALL.value == "getAll"
        assert N8nOperation.DELETE.value == "delete"
        assert N8nOperation.EXECUTE.value == "execute"


# =============================================================================
# N8nWebhook Tests
# =============================================================================


class TestN8nWebhook:
    def test_matches_event_matching(self):
        webhook = N8nWebhook(id="wh1", webhook_path="/path", events=["debate_start"])
        assert webhook.matches_event("debate_start", {}) is True

    def test_matches_event_wildcard(self):
        webhook = N8nWebhook(id="wh1", webhook_path="/path", events=["*"])
        assert webhook.matches_event("any_event", {}) is True

    def test_matches_event_no_match(self):
        webhook = N8nWebhook(id="wh1", webhook_path="/path", events=["debate_start"])
        assert webhook.matches_event("debate_end", {}) is False

    def test_matches_event_workspace_filter(self):
        webhook = N8nWebhook(id="wh1", webhook_path="/path", events=["*"], workspace_id="ws-1")
        assert webhook.matches_event("debate_start", {"workspace_id": "ws-1"}) is True
        assert webhook.matches_event("debate_start", {"workspace_id": "ws-2"}) is False


# =============================================================================
# N8nIntegration Tests
# =============================================================================


class TestN8nIntegration:
    def test_initialization(self, integration):
        assert integration.api_base == "https://test.aragora.ai"
        assert integration.is_configured is False

    def test_is_configured_with_cred(self, integration_with_cred):
        integ, _ = integration_with_cred
        assert integ.is_configured is True

    # --- Credential Management ---

    def test_create_credential(self, integration):
        cred = integration.create_credential("ws-1")
        assert cred.workspace_id == "ws-1"
        assert cred.api_key.startswith("n8n_")
        assert cred.active is True

    def test_create_credential_custom_url(self, integration):
        cred = integration.create_credential("ws-1", api_url="https://custom.api.com")
        assert cred.api_url == "https://custom.api.com"

    def test_get_credential(self, integration_with_cred):
        integ, cred = integration_with_cred
        result = integ.get_credential(cred.id)
        assert result is not None
        assert result.id == cred.id

    def test_get_credential_not_found(self, integration):
        assert integration.get_credential("bad") is None

    def test_get_credential_by_key(self, integration_with_cred):
        integ, cred = integration_with_cred
        result = integ.get_credential_by_key(cred.api_key)
        assert result is not None

    def test_get_credential_by_key_not_found(self, integration):
        assert integration.get_credential_by_key("bad") is None

    def test_list_credentials(self, integration):
        integration.create_credential("ws-1")
        integration.create_credential("ws-2")
        assert len(integration.list_credentials()) == 2
        assert len(integration.list_credentials(workspace_id="ws-1")) == 1

    def test_delete_credential(self, integration_with_cred):
        integ, cred = integration_with_cred
        # Add a webhook first to test cleanup
        integ.register_webhook(cred.id, ["debate_start"])
        assert integ.delete_credential(cred.id) is True
        assert integ.get_credential(cred.id) is None

    def test_delete_credential_not_found(self, integration):
        assert integration.delete_credential("bad") is False

    # --- Webhook Management ---

    def test_register_webhook(self, integration_with_cred):
        integ, cred = integration_with_cred
        webhook = integ.register_webhook(cred.id, ["debate_start", "debate_end"])
        assert webhook is not None
        assert webhook.events == ["debate_start", "debate_end"]
        assert webhook.webhook_path.startswith("/n8n/webhook/")

    def test_register_webhook_with_metadata(self, integration_with_cred):
        integ, cred = integration_with_cred
        webhook = integ.register_webhook(cred.id, ["*"], workflow_id="wf-1", node_id="nd-1")
        assert webhook.workflow_id == "wf-1"
        assert webhook.node_id == "nd-1"

    def test_register_webhook_invalid_cred(self, integration):
        result = integration.register_webhook("bad", ["debate_start"])
        assert result is None

    def test_register_webhook_invalid_event(self, integration_with_cred):
        integ, cred = integration_with_cred
        result = integ.register_webhook(cred.id, ["invalid_event_type"])
        assert result is None

    def test_unregister_webhook(self, integration_with_cred):
        integ, cred = integration_with_cred
        webhook = integ.register_webhook(cred.id, ["debate_start"])
        assert integ.unregister_webhook(cred.id, webhook.id) is True

    def test_unregister_webhook_not_found(self, integration_with_cred):
        integ, cred = integration_with_cred
        assert integ.unregister_webhook(cred.id, "bad") is False

    def test_unregister_webhook_bad_cred(self, integration):
        assert integration.unregister_webhook("bad", "wh1") is False

    def test_get_webhook_by_path(self, integration_with_cred):
        integ, cred = integration_with_cred
        webhook = integ.register_webhook(cred.id, ["*"])
        result = integ.get_webhook_by_path(webhook.webhook_path)
        assert result is not None
        assert result.id == webhook.id

    def test_get_webhook_by_path_not_found(self, integration):
        assert integration.get_webhook_by_path("/bad") is None

    def test_list_webhooks(self, integration_with_cred):
        integ, cred = integration_with_cred
        integ.register_webhook(cred.id, ["debate_start"])
        integ.register_webhook(cred.id, ["debate_end"])
        assert len(integ.list_webhooks(cred.id)) == 2

    def test_list_webhooks_bad_cred(self, integration):
        assert integration.list_webhooks("bad") == []

    # --- Event Dispatch ---

    @pytest.mark.asyncio
    async def test_dispatch_event(self, integration_with_cred):
        integ, cred = integration_with_cred
        integ.register_webhook(cred.id, ["debate_start"])

        with patch.object(integ, "_dispatch_to_webhook", new_callable=AsyncMock, return_value=True):
            count = await integ.dispatch_event("debate_start", {"id": "e1"})
            assert count == 1

    @pytest.mark.asyncio
    async def test_dispatch_event_no_match(self, integration_with_cred):
        integ, cred = integration_with_cred
        integ.register_webhook(cred.id, ["consensus"])
        count = await integ.dispatch_event("debate_start", {})
        assert count == 0

    # --- Node Operations ---

    @pytest.mark.asyncio
    async def test_execute_operation(self, integration_with_cred):
        integ, cred = integration_with_cred
        result = await integ.execute_operation(
            cred.id, N8nResourceType.DEBATE, N8nOperation.CREATE, {"topic": "Test"}
        )
        assert result["success"] is True
        assert result["resource"] == "debate"
        assert result["operation"] == "create"

    @pytest.mark.asyncio
    async def test_execute_operation_bad_cred(self, integration):
        result = await integration.execute_operation(
            "bad", N8nResourceType.DEBATE, N8nOperation.CREATE, {}
        )
        assert result["success"] is False

    # --- Authentication ---

    def test_authenticate_request(self, integration_with_cred):
        integ, cred = integration_with_cred
        result = integ.authenticate_request(cred.api_key)
        assert result is not None

    def test_authenticate_request_invalid(self, integration):
        assert integration.authenticate_request("bad") is None

    def test_verify_webhook_signature(self, integration_with_cred):
        integ, cred = integration_with_cred
        webhook = integ.register_webhook(cred.id, ["*"])
        payload = b'{"test": true}'
        expected = hmac.new(cred.api_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        assert integ.verify_webhook_signature(payload, expected, webhook.id) is True

    def test_verify_webhook_signature_invalid(self, integration_with_cred):
        integ, cred = integration_with_cred
        webhook = integ.register_webhook(cred.id, ["*"])
        assert integ.verify_webhook_signature(b"data", "bad_sig", webhook.id) is False

    def test_verify_webhook_signature_unknown_webhook(self, integration):
        assert integration.verify_webhook_signature(b"data", "sig", "unknown") is False

    # --- Node Definitions ---

    def test_get_node_definition(self, integration):
        defn = integration.get_node_definition()
        assert defn["name"] == "aragora"
        assert defn["version"] == 1

    def test_get_trigger_node_definition(self, integration):
        defn = integration.get_trigger_node_definition()
        assert defn["name"] == "aragoraTrigger"

    def test_get_credential_definition(self, integration):
        defn = integration.get_credential_definition()
        assert defn["name"] == "aragoraApi"
        assert len(defn["properties"]) == 2

    @pytest.mark.asyncio
    async def test_send_message_no_url(self, integration):
        result = await integration.send_message("test")
        assert result is False


# =============================================================================
# Singleton Tests
# =============================================================================


class TestGetN8nIntegration:
    def test_singleton(self):
        import aragora.integrations.n8n as mod

        mod._n8n_integration = None
        integ1 = get_n8n_integration()
        integ2 = get_n8n_integration()
        assert integ1 is integ2
        mod._n8n_integration = None
