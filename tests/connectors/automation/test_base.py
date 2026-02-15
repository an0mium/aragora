"""
Tests for automation connector base classes.

Tests AutomationEventType, WebhookSubscription, WebhookDeliveryResult,
and AutomationConnector abstract base class.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from aragora.connectors.automation.base import (
    AutomationConnector,
    AutomationEventType,
    WebhookDeliveryResult,
    WebhookSubscription,
)


# =============================================================================
# AutomationEventType Tests
# =============================================================================


class TestAutomationEventType:
    """Test AutomationEventType enum."""

    def test_debate_events_exist(self):
        assert AutomationEventType.DEBATE_STARTED == "debate.started"
        assert AutomationEventType.DEBATE_COMPLETED == "debate.completed"
        assert AutomationEventType.DEBATE_FAILED == "debate.failed"
        assert AutomationEventType.DEBATE_ROUND_COMPLETED == "debate.round_completed"

    def test_consensus_events_exist(self):
        assert AutomationEventType.CONSENSUS_REACHED == "consensus.reached"
        assert AutomationEventType.CONSENSUS_FAILED == "consensus.failed"

    def test_agent_events_exist(self):
        assert AutomationEventType.AGENT_RESPONSE == "agent.response"
        assert AutomationEventType.AGENT_CRITIQUE == "agent.critique"
        assert AutomationEventType.AGENT_VOTE == "agent.vote"

    def test_knowledge_events_exist(self):
        assert AutomationEventType.KNOWLEDGE_ADDED == "knowledge.added"
        assert AutomationEventType.KNOWLEDGE_UPDATED == "knowledge.updated"
        assert AutomationEventType.KNOWLEDGE_QUERY == "knowledge.query"

    def test_decision_events_exist(self):
        assert AutomationEventType.DECISION_MADE == "decision.made"
        assert AutomationEventType.RECEIPT_GENERATED == "receipt.generated"

    def test_system_events_exist(self):
        assert AutomationEventType.HEALTH_CHECK == "health.check"
        assert AutomationEventType.TEST_EVENT == "test.event"

    def test_is_string_enum(self):
        """Event types should be string values."""
        assert isinstance(AutomationEventType.DEBATE_STARTED, str)
        assert AutomationEventType.DEBATE_STARTED == "debate.started"

    def test_event_count(self):
        """Verify total number of event types."""
        assert len(AutomationEventType) == 18


# =============================================================================
# WebhookSubscription Tests
# =============================================================================


class TestWebhookSubscription:
    """Test WebhookSubscription dataclass."""

    def test_default_values(self):
        sub = WebhookSubscription()
        assert sub.id  # auto-generated UUID
        assert sub.webhook_url == ""
        assert sub.events == set()
        assert sub.secret  # auto-generated
        assert sub.platform == "generic"
        assert sub.enabled is True
        assert sub.verified is False
        assert sub.delivery_count == 0
        assert sub.failure_count == 0
        assert sub.retry_count == 3
        assert sub.timeout_seconds == 30

    def test_custom_values(self):
        sub = WebhookSubscription(
            webhook_url="https://example.com/hook",
            events={AutomationEventType.DEBATE_COMPLETED},
            platform="zapier",
            workspace_id="ws-1",
            user_id="usr-1",
            name="Test hook",
        )
        assert sub.webhook_url == "https://example.com/hook"
        assert AutomationEventType.DEBATE_COMPLETED in sub.events
        assert sub.platform == "zapier"
        assert sub.workspace_id == "ws-1"

    def test_to_dict(self):
        sub = WebhookSubscription(
            webhook_url="https://example.com/hook",
            events={AutomationEventType.DEBATE_COMPLETED},
            platform="zapier",
        )
        d = sub.to_dict()
        assert d["webhook_url"] == "https://example.com/hook"
        assert "debate.completed" in d["events"]
        assert d["platform"] == "zapier"
        assert d["enabled"] is True
        assert "created_at" in d

    def test_from_dict(self):
        data = {
            "id": "test-123",
            "webhook_url": "https://example.com/hook",
            "events": ["debate.completed", "consensus.reached"],
            "platform": "n8n",
            "workspace_id": "ws-1",
            "enabled": True,
            "verified": True,
            "delivery_count": 5,
            "failure_count": 1,
        }
        sub = WebhookSubscription.from_dict(data)
        assert sub.id == "test-123"
        assert sub.webhook_url == "https://example.com/hook"
        assert AutomationEventType.DEBATE_COMPLETED in sub.events
        assert AutomationEventType.CONSENSUS_REACHED in sub.events
        assert sub.platform == "n8n"
        assert sub.delivery_count == 5

    def test_roundtrip_serialization(self):
        original = WebhookSubscription(
            webhook_url="https://hooks.zapier.com/catch/123",
            events={
                AutomationEventType.DEBATE_COMPLETED,
                AutomationEventType.CONSENSUS_REACHED,
            },
            platform="zapier",
            workspace_id="ws-abc",
            name="My hook",
        )
        d = original.to_dict()
        restored = WebhookSubscription.from_dict(d)
        assert restored.webhook_url == original.webhook_url
        assert restored.events == original.events
        assert restored.platform == original.platform
        assert restored.workspace_id == original.workspace_id

    def test_unique_ids(self):
        sub1 = WebhookSubscription()
        sub2 = WebhookSubscription()
        assert sub1.id != sub2.id

    def test_unique_secrets(self):
        sub1 = WebhookSubscription()
        sub2 = WebhookSubscription()
        assert sub1.secret != sub2.secret


# =============================================================================
# WebhookDeliveryResult Tests
# =============================================================================


class TestWebhookDeliveryResult:
    """Test WebhookDeliveryResult dataclass."""

    def test_success_result(self):
        result = WebhookDeliveryResult(
            subscription_id="sub-1",
            event_type=AutomationEventType.DEBATE_COMPLETED,
            success=True,
            status_code=200,
            duration_ms=45.2,
        )
        assert result.success is True
        assert result.status_code == 200
        assert result.error is None

    def test_failure_result(self):
        result = WebhookDeliveryResult(
            subscription_id="sub-1",
            event_type=AutomationEventType.DEBATE_COMPLETED,
            success=False,
            error="Connection refused",
            duration_ms=5001.0,
        )
        assert result.success is False
        assert result.error == "Connection refused"
        assert result.status_code is None

    def test_timestamp_auto_set(self):
        result = WebhookDeliveryResult(
            subscription_id="sub-1",
            event_type=AutomationEventType.TEST_EVENT,
            success=True,
        )
        assert isinstance(result.timestamp, datetime)


# =============================================================================
# AutomationConnector Tests (via concrete subclass)
# =============================================================================


class ConcreteConnector(AutomationConnector):
    """Concrete implementation for testing abstract base class."""

    PLATFORM_NAME = "test"
    SUPPORTED_EVENTS = {
        AutomationEventType.DEBATE_COMPLETED,
        AutomationEventType.CONSENSUS_REACHED,
        AutomationEventType.TEST_EVENT,
    }

    async def format_payload(self, event_type, payload, subscription):
        return {"event": event_type.value, "data": payload}

    def generate_signature(self, payload, secret, timestamp):
        import hashlib
        import hmac as _hmac

        signed = f"{timestamp}.".encode() + payload
        return _hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


class TestAutomationConnector:
    """Test AutomationConnector base class functionality."""

    @pytest.fixture
    def connector(self):
        return ConcreteConnector()

    @pytest.mark.asyncio
    async def test_subscribe(self, connector):
        sub = await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.DEBATE_COMPLETED],
            workspace_id="ws-1",
            name="Test",
        )
        assert sub.webhook_url == "https://example.com/hook"
        assert sub.platform == "test"
        assert AutomationEventType.DEBATE_COMPLETED in sub.events

    @pytest.mark.asyncio
    async def test_subscribe_rejects_unsupported_events(self, connector):
        with pytest.raises(ValueError, match="Unsupported events"):
            await connector.subscribe(
                webhook_url="https://example.com/hook",
                events=[AutomationEventType.KNOWLEDGE_ADDED],  # Not in SUPPORTED_EVENTS
            )

    @pytest.mark.asyncio
    async def test_unsubscribe(self, connector):
        sub = await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.DEBATE_COMPLETED],
        )
        assert await connector.unsubscribe(sub.id) is True
        assert connector.get_subscription(sub.id) is None

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, connector):
        assert await connector.unsubscribe("nonexistent") is False

    @pytest.mark.asyncio
    async def test_get_subscription(self, connector):
        sub = await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.DEBATE_COMPLETED],
        )
        retrieved = connector.get_subscription(sub.id)
        assert retrieved is not None
        assert retrieved.id == sub.id

    @pytest.mark.asyncio
    async def test_list_subscriptions(self, connector):
        await connector.subscribe(
            webhook_url="https://example.com/1",
            events=[AutomationEventType.DEBATE_COMPLETED],
            workspace_id="ws-1",
        )
        await connector.subscribe(
            webhook_url="https://example.com/2",
            events=[AutomationEventType.CONSENSUS_REACHED],
            workspace_id="ws-2",
        )

        all_subs = connector.list_subscriptions()
        assert len(all_subs) == 2

        ws1_subs = connector.list_subscriptions(workspace_id="ws-1")
        assert len(ws1_subs) == 1

        debate_subs = connector.list_subscriptions(
            event_type=AutomationEventType.DEBATE_COMPLETED,
        )
        assert len(debate_subs) == 1

    @pytest.mark.asyncio
    async def test_dispatch_event_dry_run(self, connector):
        """Without http_client, dispatch runs in dry-run mode."""
        sub = await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.DEBATE_COMPLETED],
        )

        results = await connector.dispatch_event(
            AutomationEventType.DEBATE_COMPLETED,
            {"debate_id": "deb-1"},
        )
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].subscription_id == sub.id
        assert sub.delivery_count == 1

    @pytest.mark.asyncio
    async def test_dispatch_skips_disabled_subscriptions(self, connector):
        sub = await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.DEBATE_COMPLETED],
        )
        sub.enabled = False

        results = await connector.dispatch_event(
            AutomationEventType.DEBATE_COMPLETED,
            {"debate_id": "deb-1"},
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_dispatch_skips_non_matching_events(self, connector):
        await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.CONSENSUS_REACHED],
        )

        results = await connector.dispatch_event(
            AutomationEventType.DEBATE_COMPLETED,
            {"debate_id": "deb-1"},
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_dispatch_filters_by_workspace(self, connector):
        await connector.subscribe(
            webhook_url="https://example.com/1",
            events=[AutomationEventType.DEBATE_COMPLETED],
            workspace_id="ws-1",
        )
        await connector.subscribe(
            webhook_url="https://example.com/2",
            events=[AutomationEventType.DEBATE_COMPLETED],
            workspace_id="ws-2",
        )

        results = await connector.dispatch_event(
            AutomationEventType.DEBATE_COMPLETED,
            {"debate_id": "deb-1"},
            workspace_id="ws-1",
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatch_with_http_client(self, connector):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"ok": true}'

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        connector._http_client = mock_client

        sub = await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.DEBATE_COMPLETED],
        )

        results = await connector.dispatch_event(
            AutomationEventType.DEBATE_COMPLETED,
            {"debate_id": "deb-1"},
        )
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].status_code == 200
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_http_failure(self, connector):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        connector._http_client = mock_client

        sub = await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.DEBATE_COMPLETED],
        )

        results = await connector.dispatch_event(
            AutomationEventType.DEBATE_COMPLETED,
            {"debate_id": "deb-1"},
        )
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].status_code == 500
        assert sub.failure_count == 1

    @pytest.mark.asyncio
    async def test_dispatch_connection_error(self, connector):
        mock_client = AsyncMock()
        mock_client.post.side_effect = OSError("Connection refused")
        connector._http_client = mock_client

        sub = await connector.subscribe(
            webhook_url="https://example.com/hook",
            events=[AutomationEventType.DEBATE_COMPLETED],
        )

        results = await connector.dispatch_event(
            AutomationEventType.DEBATE_COMPLETED,
            {"debate_id": "deb-1"},
        )
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error  # Sanitized error message present
        assert "failed" in results[0].error.lower()
        assert sub.failure_count == 1

    def test_verify_signature(self, connector):
        payload = b'{"test": true}'
        secret = "test-secret-key"
        timestamp = 1700000000

        sig = connector.generate_signature(payload, secret, timestamp)
        assert connector.verify_signature(payload, sig, secret, timestamp) is True

    def test_verify_signature_wrong_secret(self, connector):
        payload = b'{"test": true}'
        timestamp = 1700000000

        sig = connector.generate_signature(payload, "correct-secret", timestamp)
        assert connector.verify_signature(payload, sig, "wrong-secret", timestamp) is False

    def test_verify_signature_wrong_timestamp(self, connector):
        payload = b'{"test": true}'
        secret = "test-secret"

        sig = connector.generate_signature(payload, secret, 1700000000)
        assert connector.verify_signature(payload, sig, secret, 1700000001) is False
