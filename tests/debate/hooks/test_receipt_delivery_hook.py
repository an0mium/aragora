"""Tests for Receipt Delivery Hook."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.hooks.receipt_delivery_hook import (
    DeliveryResult,
    ReceiptDeliveryHook,
    create_receipt_delivery_hook,
)


@dataclass
class MockDebateContext:
    """Mock debate context for testing."""

    debate_id: str = "debate-123"
    task: str = "Test debate task"
    org_id: str = "org-456"


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    id: str = "result-123"
    debate_id: str = "debate-123"
    task: str = "Test debate task"
    final_answer: str = "The consensus answer is 42."
    confidence: float = 0.85
    consensus_reached: bool = True
    rounds_used: int = 3
    rounds_completed: int = 3
    participants: list[str] = field(default_factory=lambda: ["agent-1", "agent-2"])
    duration_seconds: float = 120.5


@dataclass
class MockChannelSubscription:
    """Mock channel subscription for testing."""

    id: str = "sub-123"
    org_id: str = "org-456"
    channel_type: str = "slack"
    channel_id: str = "C123456"
    event_types: list[str] = field(default_factory=lambda: ["receipt"])
    workspace_id: str | None = "T123456"
    is_active: bool = True


@dataclass
class MockSlackWorkspace:
    """Mock Slack workspace for testing."""

    workspace_id: str = "T123456"
    access_token: str = "xoxb-test-token"
    signing_secret: str = "secret123"


@dataclass
class MockTeamsWorkspace:
    """Mock Teams workspace for testing."""

    tenant_id: str = "tenant-123"
    bot_id: str = "bot-123"
    service_url: str = "https://smba.trafficmanager.net/amer/"


@dataclass
class MockSendResult:
    """Mock send result for testing."""

    timestamp: str = "1234567890.123456"
    channel_id: str = "C123456"
    message_id: str = "msg-123"


# ============================================================================
# Factory Tests
# ============================================================================


class TestCreateReceiptDeliveryHook:
    """Tests for the factory function."""

    def test_create_with_defaults(self):
        """Test creating hook with default settings."""
        hook = create_receipt_delivery_hook(org_id="org-123")

        assert hook.org_id == "org-123"
        assert hook.min_confidence == 0.0
        assert hook.require_consensus is False
        assert hook.enabled is True

    def test_create_with_custom_settings(self):
        """Test creating hook with custom settings."""
        hook = create_receipt_delivery_hook(
            org_id="org-456",
            min_confidence=0.7,
            require_consensus=True,
            enabled=False,
        )

        assert hook.org_id == "org-456"
        assert hook.min_confidence == 0.7
        assert hook.require_consensus is True
        assert hook.enabled is False


# ============================================================================
# Delivery Result Tests
# ============================================================================


class TestDeliveryResult:
    """Tests for DeliveryResult dataclass."""

    def test_to_dict_success(self):
        """Test converting successful delivery to dict."""
        result = DeliveryResult(
            channel_type="slack",
            channel_id="C123456",
            workspace_id="T123456",
            success=True,
            message_id="1234567890.123456",
        )

        d = result.to_dict()

        assert d["channel_type"] == "slack"
        assert d["channel_id"] == "C123456"
        assert d["workspace_id"] == "T123456"
        assert d["success"] is True
        assert d["message_id"] == "1234567890.123456"
        assert d["error"] is None
        assert "timestamp" in d
        assert "timestamp_iso" in d

    def test_to_dict_failure(self):
        """Test converting failed delivery to dict."""
        result = DeliveryResult(
            channel_type="teams",
            channel_id="channel-123",
            workspace_id="tenant-456",
            success=False,
            error="Connection refused",
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "Connection refused"


# ============================================================================
# Hook Lifecycle Tests
# ============================================================================


class TestHookLifecycle:
    """Tests for hook enable/disable behavior."""

    @pytest.mark.asyncio
    async def test_disabled_hook_skips_delivery(self):
        """Test that disabled hook does not deliver."""
        hook = create_receipt_delivery_hook(org_id="org-123", enabled=False)

        ctx = MockDebateContext()
        result = MockDebateResult()

        # Should not raise or attempt delivery
        await hook.on_post_debate(ctx, result)

        assert len(hook.get_delivery_history()) == 0

    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self):
        """Test that low confidence results are filtered."""
        hook = create_receipt_delivery_hook(org_id="org-123", min_confidence=0.9, enabled=True)

        ctx = MockDebateContext()
        result = MockDebateResult(confidence=0.5)  # Below threshold

        # Mock to avoid actual delivery
        with patch.object(hook, "_get_receipt_subscriptions", return_value=[]):
            await hook.on_post_debate(ctx, result)

        # Should not attempt delivery due to low confidence
        assert len(hook.get_delivery_history()) == 0

    @pytest.mark.asyncio
    async def test_consensus_requirement_filtering(self):
        """Test that non-consensus results are filtered when required."""
        hook = create_receipt_delivery_hook(org_id="org-123", require_consensus=True, enabled=True)

        ctx = MockDebateContext()
        result = MockDebateResult(consensus_reached=False)

        with patch.object(hook, "_get_receipt_subscriptions", return_value=[]):
            await hook.on_post_debate(ctx, result)

        assert len(hook.get_delivery_history()) == 0

    @pytest.mark.asyncio
    async def test_no_subscriptions_no_delivery(self):
        """Test that no delivery occurs when no subscriptions exist."""
        hook = create_receipt_delivery_hook(org_id="org-123", enabled=True)

        ctx = MockDebateContext()
        result = MockDebateResult()

        with patch.object(hook, "_get_receipt_subscriptions", return_value=[]):
            await hook.on_post_debate(ctx, result)

        assert len(hook.get_delivery_history()) == 0


# ============================================================================
# Delivery Tests
# ============================================================================


class TestSlackDelivery:
    """Tests for Slack delivery."""

    @pytest.mark.asyncio
    async def test_slack_delivery_success(self):
        """Test successful Slack delivery."""
        hook = create_receipt_delivery_hook(org_id="org-456")
        receipt = {"debate_id": "debate-123", "task": "Test", "confidence": 0.9}

        # Mock workspace store at the storage module level
        mock_workspace = MockSlackWorkspace()
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store"
        ) as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get.return_value = mock_workspace
            mock_store_fn.return_value = mock_store

            # Mock connector
            with patch("aragora.connectors.chat.slack.SlackConnector") as MockConnector:
                mock_connector = MagicMock()
                mock_connector.send_message = AsyncMock(return_value=MockSendResult())
                MockConnector.return_value = mock_connector

                result = await hook._send_to_slack(receipt, "C123456", "T123456")

        assert result.success is True
        assert result.channel_type == "slack"

    @pytest.mark.asyncio
    async def test_slack_delivery_no_workspace_id(self):
        """Test Slack delivery fails without workspace_id."""
        hook = create_receipt_delivery_hook(org_id="org-456")
        receipt = {"debate_id": "debate-123"}

        result = await hook._send_to_slack(receipt, "C123456", None)

        assert result.success is False
        assert "workspace_id is required" in result.error

    @pytest.mark.asyncio
    async def test_slack_delivery_workspace_not_found(self):
        """Test Slack delivery fails when workspace not found."""
        hook = create_receipt_delivery_hook(org_id="org-456")
        receipt = {"debate_id": "debate-123"}

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store"
        ) as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get.return_value = None
            mock_store_fn.return_value = mock_store

            result = await hook._send_to_slack(receipt, "C123456", "T-invalid")

        assert result.success is False
        assert "not found" in result.error


class TestTeamsDelivery:
    """Tests for Teams delivery."""

    @pytest.mark.asyncio
    async def test_teams_delivery_success(self):
        """Test successful Teams delivery."""
        hook = create_receipt_delivery_hook(org_id="org-456")
        receipt = {"debate_id": "debate-123", "task": "Test", "confidence": 0.9}

        mock_workspace = MockTeamsWorkspace()
        with patch(
            "aragora.storage.teams_workspace_store.get_teams_workspace_store"
        ) as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get.return_value = mock_workspace
            mock_store_fn.return_value = mock_store

            with patch("aragora.connectors.chat.teams.TeamsConnector") as MockConnector:
                mock_connector = MagicMock()
                mock_connector.send_message = AsyncMock(return_value=MockSendResult())
                MockConnector.return_value = mock_connector

                result = await hook._send_to_teams(receipt, "channel-123", "tenant-123")

        assert result.success is True
        assert result.channel_type == "teams"

    @pytest.mark.asyncio
    async def test_teams_delivery_no_workspace_id(self):
        """Test Teams delivery fails without workspace_id."""
        hook = create_receipt_delivery_hook(org_id="org-456")
        receipt = {"debate_id": "debate-123"}

        result = await hook._send_to_teams(receipt, "channel-123", None)

        assert result.success is False
        assert "workspace_id" in result.error


class TestWebhookDelivery:
    """Tests for webhook delivery."""

    @pytest.mark.asyncio
    async def test_webhook_delivery_success(self):
        """Test successful webhook delivery."""
        hook = create_receipt_delivery_hook(org_id="org-456")
        receipt = {"debate_id": "debate-123"}

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await hook._send_to_webhook(receipt, "https://example.com/webhook")

        assert result.success is True
        assert result.channel_type == "webhook"

    @pytest.mark.asyncio
    async def test_webhook_delivery_failure(self):
        """Test webhook delivery failure."""
        hook = create_receipt_delivery_hook(org_id="org-456")
        receipt = {"debate_id": "debate-123"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await hook._send_to_webhook(receipt, "https://example.com/webhook")

        assert result.success is False
        assert "delivery_failed:webhook" in result.error


# ============================================================================
# Formatting Tests
# ============================================================================


class TestReceiptFormatting:
    """Tests for receipt formatting."""

    def test_format_for_slack(self):
        """Test formatting receipt for Slack."""
        hook = create_receipt_delivery_hook(org_id="org-123")
        receipt = {
            "debate_id": "debate-123",
            "task": "Test task",
            "final_answer": "The answer is 42",
            "confidence": 0.85,
            "consensus_reached": True,
            "rounds_used": 3,
        }

        blocks = hook._format_receipt_for_slack(receipt)

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"
        assert "Decision Receipt" in blocks[0]["text"]["text"]

    def test_format_for_teams(self):
        """Test formatting receipt for Teams."""
        hook = create_receipt_delivery_hook(org_id="org-123")
        receipt = {
            "debate_id": "debate-123",
            "task": "Test task",
            "final_answer": "The answer is 42",
            "confidence": 0.85,
            "consensus_reached": True,
            "rounds_used": 3,
        }

        card_body = hook._format_receipt_for_teams(receipt)

        assert len(card_body) > 0
        assert card_body[0]["type"] == "TextBlock"
        assert "Decision Receipt" in card_body[0]["text"]

    def test_format_for_email(self):
        """Test formatting receipt for email."""
        hook = create_receipt_delivery_hook(org_id="org-123")
        receipt = {
            "debate_id": "debate-123",
            "task": "Test task",
            "final_answer": "The answer is 42",
            "confidence": 0.85,
            "consensus_reached": True,
            "rounds_used": 3,
        }

        html, plain = hook._format_receipt_for_email(receipt)

        assert "Decision Receipt" in html
        assert "Decision Receipt" in plain
        assert "Test task" in html
        assert "Test task" in plain


# ============================================================================
# Integration Tests
# ============================================================================


class TestFullDeliveryFlow:
    """Integration tests for full delivery flow."""

    @pytest.mark.asyncio
    async def test_full_delivery_to_multiple_channels(self):
        """Test delivering to multiple channels."""
        hook = create_receipt_delivery_hook(org_id="org-456")

        ctx = MockDebateContext()
        result = MockDebateResult()

        # Mock subscriptions
        subscriptions = [
            MockChannelSubscription(id="sub-1", channel_type="slack"),
            MockChannelSubscription(id="sub-2", channel_type="teams", workspace_id="tenant-123"),
        ]

        with patch.object(hook, "_get_receipt_subscriptions", return_value=subscriptions):
            with patch.object(hook, "_generate_receipt") as mock_gen:
                mock_gen.return_value = {
                    "debate_id": "debate-123",
                    "task": "Test",
                    "confidence": 0.9,
                }

                # Mock both delivery methods
                with patch.object(hook, "_send_to_slack") as mock_slack:
                    with patch.object(hook, "_send_to_teams") as mock_teams:
                        mock_slack.return_value = DeliveryResult(
                            channel_type="slack",
                            channel_id="C123456",
                            workspace_id="T123456",
                            success=True,
                        )
                        mock_teams.return_value = DeliveryResult(
                            channel_type="teams",
                            channel_id="channel-123",
                            workspace_id="tenant-123",
                            success=True,
                        )

                        await hook.on_post_debate(ctx, result)

        # Should have 2 delivery results in history
        history = hook.get_delivery_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_partial_delivery_failure(self):
        """Test handling partial delivery failures."""
        hook = create_receipt_delivery_hook(org_id="org-456")

        ctx = MockDebateContext()
        result = MockDebateResult()

        subscriptions = [
            MockChannelSubscription(id="sub-1", channel_type="slack"),
            MockChannelSubscription(
                id="sub-2", channel_type="webhook", channel_id="https://bad.url"
            ),
        ]

        with patch.object(hook, "_get_receipt_subscriptions", return_value=subscriptions):
            with patch.object(hook, "_generate_receipt") as mock_gen:
                mock_gen.return_value = {"debate_id": "debate-123"}

                with patch.object(hook, "_send_to_slack") as mock_slack:
                    with patch.object(hook, "_send_to_webhook") as mock_webhook:
                        mock_slack.return_value = DeliveryResult(
                            channel_type="slack",
                            channel_id="C123456",
                            workspace_id="T123456",
                            success=True,
                        )
                        mock_webhook.return_value = DeliveryResult(
                            channel_type="webhook",
                            channel_id="https://bad.url",
                            workspace_id=None,
                            success=False,
                            error="Connection refused",
                        )

                        await hook.on_post_debate(ctx, result)

        history = hook.get_delivery_history()
        assert len(history) == 2

        success_count = sum(1 for h in history if h.success)
        assert success_count == 1


# ============================================================================
# Receipt Generation Tests
# ============================================================================


class TestReceiptGeneration:
    """Tests for receipt generation."""

    @pytest.mark.asyncio
    async def test_generate_receipt_basic(self):
        """Test basic receipt generation."""
        hook = create_receipt_delivery_hook(org_id="org-456")

        ctx = MockDebateContext()
        result = MockDebateResult()

        receipt = await hook._generate_receipt(ctx, result)

        assert receipt is not None
        assert "debate_id" in receipt
        assert "task" in receipt
        assert "confidence" in receipt
        assert "consensus_reached" in receipt
        assert "content_hash" in receipt
        assert "org_id" in receipt
        assert receipt["org_id"] == "org-456"

    @pytest.mark.asyncio
    async def test_generate_receipt_includes_hash(self):
        """Test that receipt includes content hash for integrity."""
        hook = create_receipt_delivery_hook(org_id="org-456")

        ctx = MockDebateContext()
        result = MockDebateResult()

        receipt = await hook._generate_receipt(ctx, result)

        assert receipt is not None
        assert "content_hash" in receipt
        # Should be a valid SHA-256 hex string (64 chars)
        assert len(receipt["content_hash"]) == 64


# ============================================================================
# Subscription Tests
# ============================================================================


class TestSubscriptionRetrieval:
    """Tests for subscription retrieval."""

    @pytest.mark.asyncio
    async def test_get_subscriptions_success(self):
        """Test successful subscription retrieval."""
        hook = create_receipt_delivery_hook(org_id="org-456")

        mock_subs = [MockChannelSubscription()]

        with patch(
            "aragora.storage.channel_subscription_store.get_channel_subscription_store"
        ) as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get_by_org.return_value = mock_subs
            mock_store_fn.return_value = mock_store

            subs = await hook._get_receipt_subscriptions()

        assert len(subs) == 1

    @pytest.mark.asyncio
    async def test_get_subscriptions_filters_inactive(self):
        """Test that inactive subscriptions are filtered."""
        hook = create_receipt_delivery_hook(org_id="org-456")

        mock_subs = [
            MockChannelSubscription(id="sub-1", is_active=True),
            MockChannelSubscription(id="sub-2", is_active=False),
        ]

        with patch(
            "aragora.storage.channel_subscription_store.get_channel_subscription_store"
        ) as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get_by_org.return_value = mock_subs
            mock_store_fn.return_value = mock_store

            subs = await hook._get_receipt_subscriptions()

        # Should only return active subscription
        assert len(subs) == 1
        assert subs[0].id == "sub-1"

    @pytest.mark.asyncio
    async def test_get_subscriptions_import_error(self):
        """Test handling import error for subscription store."""
        hook = create_receipt_delivery_hook(org_id="org-456")

        # Force import to fail by patching sys.modules
        with patch.dict("sys.modules", {"aragora.storage.channel_subscription_store": None}):
            subs = await hook._get_receipt_subscriptions()

        assert subs == []
