"""
Tests for Zapier automation connector.
"""

import hashlib
import hmac
import json

import pytest
from datetime import datetime, timezone

from aragora.connectors.automation.base import (
    AutomationEventType,
    WebhookSubscription,
)
from aragora.connectors.automation.zapier import ZapierConnector


class TestZapierConnector:
    """Test ZapierConnector."""

    @pytest.fixture
    def connector(self):
        return ZapierConnector()

    def test_platform_name(self, connector):
        assert connector.PLATFORM_NAME == "zapier"

    def test_supports_all_events(self, connector):
        assert connector.SUPPORTED_EVENTS == set(AutomationEventType)

    def test_max_payload_size(self, connector):
        assert connector.MAX_PAYLOAD_SIZE == 6 * 1024 * 1024


class TestZapierPayloadFormatting:
    """Test Zapier payload formatting."""

    @pytest.fixture
    def connector(self):
        return ZapierConnector()

    @pytest.fixture
    def subscription(self):
        return WebhookSubscription(
            id="sub-123",
            webhook_url="https://hooks.zapier.com/catch/123/abc",
            events={AutomationEventType.DEBATE_COMPLETED},
            platform="zapier",
            workspace_id="ws-1",
        )

    @pytest.mark.asyncio
    async def test_format_payload_envelope(self, connector, subscription):
        payload = {"debate_id": "deb-1", "consensus": True}
        formatted = await connector.format_payload(
            AutomationEventType.DEBATE_COMPLETED,
            payload,
            subscription,
        )

        assert formatted["event_type"] == "debate.completed"
        assert formatted["event_category"] == "debate"
        assert formatted["subscription_id"] == "sub-123"
        assert formatted["workspace_id"] == "ws-1"
        assert "timestamp" in formatted
        assert "id" in formatted

    @pytest.mark.asyncio
    async def test_format_payload_flattens_nested(self, connector, subscription):
        payload = {
            "debate": {
                "id": "deb-1",
                "result": {
                    "consensus": True,
                },
            },
        }
        formatted = await connector.format_payload(
            AutomationEventType.DEBATE_COMPLETED,
            payload,
            subscription,
        )

        # Nested keys should be flattened with underscore separator
        assert formatted["debate_id"] == "deb-1"
        assert formatted["debate_result_consensus"] is True

    @pytest.mark.asyncio
    async def test_format_payload_handles_arrays(self, connector, subscription):
        payload = {
            "agents": ["claude", "gpt-4", "gemini"],
        }
        formatted = await connector.format_payload(
            AutomationEventType.DEBATE_COMPLETED,
            payload,
            subscription,
        )

        assert formatted["agents"] == ["claude", "gpt-4", "gemini"]
        assert formatted["agents_count"] == 3


class TestZapierFlatten:
    """Test Zapier payload flattening."""

    @pytest.fixture
    def connector(self):
        return ZapierConnector()

    def test_flatten_simple(self, connector):
        data = {"key": "value", "num": 42}
        result = connector._flatten_for_zapier(data)
        assert result == {"key": "value", "num": 42}

    def test_flatten_nested(self, connector):
        data = {"outer": {"inner": "value"}}
        result = connector._flatten_for_zapier(data)
        assert result == {"outer_inner": "value"}

    def test_flatten_deep_nested(self, connector):
        data = {"a": {"b": {"c": "deep"}}}
        result = connector._flatten_for_zapier(data)
        assert result == {"a_b_c": "deep"}

    def test_flatten_with_list(self, connector):
        data = {"items": [1, 2, 3]}
        result = connector._flatten_for_zapier(data)
        assert result["items"] == [1, 2, 3]
        assert result["items_count"] == 3

    def test_flatten_empty_dict(self, connector):
        result = connector._flatten_for_zapier({})
        assert result == {}

    def test_flatten_with_prefix(self, connector):
        data = {"key": "value"}
        result = connector._flatten_for_zapier(data, prefix="prefix_")
        assert result == {"prefix_key": "value"}


class TestZapierSignature:
    """Test Zapier webhook signature generation."""

    @pytest.fixture
    def connector(self):
        return ZapierConnector()

    def test_generate_signature(self, connector):
        payload = b'{"test": true}'
        secret = "my-secret"
        timestamp = 1700000000

        sig = connector.generate_signature(payload, secret, timestamp)

        # Verify manually
        signed_payload = f"{timestamp}.".encode() + payload
        expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
        assert sig == expected

    def test_signature_changes_with_different_payload(self, connector):
        secret = "my-secret"
        timestamp = 1700000000

        sig1 = connector.generate_signature(b'{"a": 1}', secret, timestamp)
        sig2 = connector.generate_signature(b'{"a": 2}', secret, timestamp)
        assert sig1 != sig2

    def test_signature_changes_with_different_timestamp(self, connector):
        payload = b'{"test": true}'
        secret = "my-secret"

        sig1 = connector.generate_signature(payload, secret, 1700000000)
        sig2 = connector.generate_signature(payload, secret, 1700000001)
        assert sig1 != sig2

    def test_verify_valid_signature(self, connector):
        payload = b'{"test": true}'
        secret = "my-secret"
        timestamp = 1700000000

        sig = connector.generate_signature(payload, secret, timestamp)
        assert connector.verify_signature(payload, sig, secret, timestamp) is True

    def test_verify_invalid_signature(self, connector):
        payload = b'{"test": true}'
        secret = "my-secret"
        timestamp = 1700000000

        assert connector.verify_signature(payload, "invalid-sig", secret, timestamp) is False


class TestZapierTestSubscription:
    """Test Zapier subscription testing."""

    @pytest.fixture
    def connector(self):
        return ZapierConnector()

    @pytest.mark.asyncio
    async def test_test_subscription_dry_run(self, connector):
        """Without http_client, test runs in dry-run mode."""
        sub = await connector.subscribe(
            webhook_url="https://hooks.zapier.com/catch/123",
            events=[AutomationEventType.TEST_EVENT],
        )

        result = await connector.test_subscription(sub)
        assert result is True
        assert sub.verified is True


class TestZapierSampleData:
    """Test Zapier sample data generation."""

    @pytest.fixture
    def connector(self):
        return ZapierConnector()

    @pytest.mark.asyncio
    async def test_debate_completed_sample(self, connector):
        sample = await connector.get_sample_data(AutomationEventType.DEBATE_COMPLETED)
        assert "debate_id" in sample
        assert "task" in sample
        assert "confidence" in sample

    @pytest.mark.asyncio
    async def test_consensus_reached_sample(self, connector):
        sample = await connector.get_sample_data(AutomationEventType.CONSENSUS_REACHED)
        assert "consensus_type" in sample
        assert "supporting_agents" in sample

    @pytest.mark.asyncio
    async def test_knowledge_added_sample(self, connector):
        sample = await connector.get_sample_data(AutomationEventType.KNOWLEDGE_ADDED)
        assert "knowledge_id" in sample
        assert "title" in sample

    @pytest.mark.asyncio
    async def test_decision_made_sample(self, connector):
        sample = await connector.get_sample_data(AutomationEventType.DECISION_MADE)
        assert "decision_id" in sample
        assert "decision" in sample

    @pytest.mark.asyncio
    async def test_unknown_event_returns_default(self, connector):
        sample = await connector.get_sample_data(AutomationEventType.HEALTH_CHECK)
        assert sample["sample"] is True
        assert "health.check" in sample["event_type"]
