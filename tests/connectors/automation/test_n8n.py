"""
Tests for n8n automation connector.
"""

import hashlib
import hmac

import pytest

from aragora.connectors.automation.base import (
    AutomationEventType,
    WebhookSubscription,
)
from aragora.connectors.automation.n8n import N8NConnector


class TestN8NConnector:
    """Test N8NConnector."""

    @pytest.fixture
    def connector(self):
        return N8NConnector()

    def test_platform_name(self, connector):
        assert connector.PLATFORM_NAME == "n8n"

    def test_supports_all_events(self, connector):
        assert connector.SUPPORTED_EVENTS == set(AutomationEventType)

    def test_default_base_url(self, connector):
        assert connector.aragora_base_url == "http://localhost:8080"

    def test_custom_base_url(self):
        connector = N8NConnector(aragora_base_url="https://api.aragora.ai")
        assert connector.aragora_base_url == "https://api.aragora.ai"


class TestN8NPayloadFormatting:
    """Test n8n payload formatting."""

    @pytest.fixture
    def connector(self):
        return N8NConnector()

    @pytest.fixture
    def subscription(self):
        return WebhookSubscription(
            id="sub-456",
            webhook_url="http://localhost:5678/webhook/aragora",
            events={AutomationEventType.DEBATE_COMPLETED},
            platform="n8n",
            workspace_id="ws-1",
            user_id="usr-1",
        )

    @pytest.mark.asyncio
    async def test_format_payload_structure(self, connector, subscription):
        payload = {"debate_id": "deb-1", "consensus": True}
        formatted = await connector.format_payload(
            AutomationEventType.DEBATE_COMPLETED,
            payload,
            subscription,
        )

        # n8n uses structured format with meta, context, data
        assert "meta" in formatted
        assert "context" in formatted
        assert "data" in formatted

    @pytest.mark.asyncio
    async def test_format_payload_meta(self, connector, subscription):
        payload = {"debate_id": "deb-1"}
        formatted = await connector.format_payload(
            AutomationEventType.DEBATE_COMPLETED,
            payload,
            subscription,
        )

        meta = formatted["meta"]
        assert meta["event_type"] == "debate.completed"
        assert meta["category"] == "debate"
        assert meta["action"] == "completed"
        assert meta["source"] == "aragora"
        assert "event_id" in meta
        assert "timestamp" in meta

    @pytest.mark.asyncio
    async def test_format_payload_context(self, connector, subscription):
        payload = {}
        formatted = await connector.format_payload(
            AutomationEventType.DEBATE_COMPLETED,
            payload,
            subscription,
        )

        context = formatted["context"]
        assert context["subscription_id"] == "sub-456"
        assert context["workspace_id"] == "ws-1"
        assert context["user_id"] == "usr-1"

    @pytest.mark.asyncio
    async def test_format_payload_preserves_data(self, connector, subscription):
        payload = {
            "debate_id": "deb-1",
            "nested": {"key": "value"},
            "agents": ["claude", "gpt-4"],
        }
        formatted = await connector.format_payload(
            AutomationEventType.DEBATE_COMPLETED,
            payload,
            subscription,
        )

        # n8n preserves data structure (unlike Zapier which flattens)
        assert formatted["data"] == payload
        assert formatted["data"]["nested"]["key"] == "value"


class TestN8NSignature:
    """Test n8n webhook signature generation."""

    @pytest.fixture
    def connector(self):
        return N8NConnector()

    def test_generate_signature_format(self, connector):
        payload = b'{"test": true}'
        secret = "my-secret"
        timestamp = 1700000000

        sig = connector.generate_signature(payload, secret, timestamp)

        # n8n uses v0= prefix
        assert sig.startswith("v0=")

    def test_generate_signature_correct(self, connector):
        payload = b'{"test": true}'
        secret = "my-secret"
        timestamp = 1700000000

        sig = connector.generate_signature(payload, secret, timestamp)

        # Verify manually
        signed_payload = f"v0:{timestamp}:".encode() + payload
        expected_hash = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
        assert sig == f"v0={expected_hash}"

    def test_signature_changes_with_payload(self, connector):
        secret = "my-secret"
        timestamp = 1700000000

        sig1 = connector.generate_signature(b'{"a": 1}', secret, timestamp)
        sig2 = connector.generate_signature(b'{"a": 2}', secret, timestamp)
        assert sig1 != sig2

    def test_verify_valid_signature(self, connector):
        payload = b'{"test": true}'
        secret = "my-secret"
        timestamp = 1700000000

        sig = connector.generate_signature(payload, secret, timestamp)
        assert connector.verify_signature(payload, sig, secret, timestamp) is True

    def test_verify_invalid_signature(self, connector):
        payload = b'{"test": true}'
        assert connector.verify_signature(payload, "v0=invalid", "secret", 1700000000) is False


class TestN8NNodeDefinition:
    """Test n8n node definition generation."""

    @pytest.fixture
    def connector(self):
        return N8NConnector()

    def test_node_definition_structure(self, connector):
        node_def = connector.get_node_definition()
        assert node_def["name"] == "Aragora"
        assert node_def["displayName"] == "Aragora"
        assert node_def["version"] == 1
        assert "inputs" in node_def
        assert "outputs" in node_def
        assert "credentials" in node_def
        assert "properties" in node_def

    def test_node_definition_credentials(self, connector):
        node_def = connector.get_node_definition()
        creds = node_def["credentials"]
        assert len(creds) == 1
        assert creds[0]["name"] == "aragoraApi"
        assert creds[0]["required"] is True

    def test_node_properties_resources(self, connector):
        node_def = connector.get_node_definition()
        props = node_def["properties"]

        # First property should be Resource selector
        resource_prop = props[0]
        assert resource_prop["name"] == "resource"
        resource_names = [opt["value"] for opt in resource_prop["options"]]
        assert "debate" in resource_names
        assert "knowledge" in resource_names
        assert "agent" in resource_names
        assert "decision" in resource_names

    def test_node_properties_debate_operations(self, connector):
        props = connector._get_node_properties()

        # Find debate operations
        debate_ops = None
        for prop in props:
            if prop.get("name") == "operation" and prop.get("displayOptions", {}).get(
                "show", {}
            ).get("resource") == ["debate"]:
                debate_ops = prop
                break

        assert debate_ops is not None
        op_values = [opt["value"] for opt in debate_ops["options"]]
        assert "start" in op_values
        assert "status" in op_values
        assert "result" in op_values
        assert "list" in op_values


class TestN8NCredentialsDefinition:
    """Test n8n credentials definition."""

    @pytest.fixture
    def connector(self):
        return N8NConnector(aragora_base_url="https://api.aragora.ai")

    def test_credentials_structure(self, connector):
        creds = connector.get_credentials_definition()
        assert creds["name"] == "aragoraApi"
        assert creds["displayName"] == "Aragora API"
        assert "properties" in creds
        assert "authenticate" in creds

    def test_credentials_properties(self, connector):
        creds = connector.get_credentials_definition()
        props = creds["properties"]
        prop_names = [p["name"] for p in props]
        assert "apiUrl" in prop_names
        assert "apiToken" in prop_names

    def test_credentials_default_url(self, connector):
        creds = connector.get_credentials_definition()
        url_prop = next(p for p in creds["properties"] if p["name"] == "apiUrl")
        assert url_prop["default"] == "https://api.aragora.ai"

    def test_credentials_auth_type(self, connector):
        creds = connector.get_credentials_definition()
        assert creds["authenticate"]["type"] == "generic"


class TestN8NTriggerDefinition:
    """Test n8n trigger node definition."""

    @pytest.fixture
    def connector(self):
        return N8NConnector()

    def test_trigger_structure(self, connector):
        trigger = connector.get_trigger_definition()
        assert trigger["name"] == "AragoraTrigger"
        assert trigger["group"] == ["trigger"]
        assert trigger["inputs"] == []  # Triggers have no inputs
        assert trigger["outputs"] == ["main"]

    def test_trigger_webhook_config(self, connector):
        trigger = connector.get_trigger_definition()
        webhooks = trigger["webhooks"]
        assert len(webhooks) == 1
        assert webhooks[0]["httpMethod"] == "POST"
        assert webhooks[0]["path"] == "aragora"

    def test_trigger_event_options(self, connector):
        trigger = connector.get_trigger_definition()
        events_prop = trigger["properties"][0]
        assert events_prop["name"] == "events"
        assert events_prop["type"] == "multiOptions"

        event_values = [opt["value"] for opt in events_prop["options"]]
        assert "debate.completed" in event_values
        assert "consensus.reached" in event_values
        assert "decision.made" in event_values
        assert "knowledge.added" in event_values
