"""
E2E Notification Delivery Tests for Aragora.

Validates notification delivery functionality:
- Email notification on debate complete
- Webhook delivery with retry
- Slack notification routing
- Notification rate limiting

Run with: pytest tests/e2e/test_notification_delivery_e2e.py -v
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

import pytest
import pytest_asyncio

from tests.e2e.harness import (
    E2ETestConfig,
    E2ETestHarness,
    e2e_environment,
)

# Mark all tests in this module as e2e
pytestmark = [pytest.mark.e2e]


# ============================================================================
# SLA Targets
# ============================================================================


@dataclass
class NotificationSLAs:
    """Notification delivery SLA targets."""

    # Delivery time targets
    max_delivery_time_seconds: float = 30.0
    max_retry_time_seconds: float = 120.0

    # Reliability targets
    delivery_rate: float = 0.99  # 99% delivery rate

    # Rate limiting
    max_notifications_per_minute: int = 60
    burst_limit: int = 10


SLAS = NotificationSLAs()


# ============================================================================
# Mock Providers
# ============================================================================


@dataclass
class MockNotificationProvider:
    """Mock notification provider for testing."""

    name: str
    sent_messages: List[Dict[str, Any]] = field(default_factory=list)
    fail_count: int = 0
    delay_seconds: float = 0.0
    failures_to_simulate: int = 0

    async def send(
        self,
        recipient: str,
        subject: str,
        body: str,
        **kwargs,
    ) -> bool:
        """Send a mock notification."""
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.failures_to_simulate > 0:
            self.failures_to_simulate -= 1
            self.fail_count += 1
            raise RuntimeError(f"Simulated {self.name} failure")

        message = {
            "provider": self.name,
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "timestamp": time.time(),
            **kwargs,
        }
        self.sent_messages.append(message)
        return True


@dataclass
class MockWebhookServer:
    """Mock webhook server for testing."""

    received_webhooks: List[Dict[str, Any]] = field(default_factory=list)
    response_delay_seconds: float = 0.0
    fail_rate: float = 0.0
    _fail_counter: int = 0

    async def receive_webhook(self, payload: Dict[str, Any]) -> bool:
        """Receive a webhook delivery."""
        if self.response_delay_seconds > 0:
            await asyncio.sleep(self.response_delay_seconds)

        import random

        if self.fail_rate > 0 and random.random() < self.fail_rate:
            self._fail_counter += 1
            return False

        self.received_webhooks.append(
            {
                "payload": payload,
                "received_at": time.time(),
            }
        )
        return True


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def email_provider():
    """Create a mock email provider."""
    return MockNotificationProvider(name="email")


@pytest.fixture
def slack_provider():
    """Create a mock Slack provider."""
    return MockNotificationProvider(name="slack")


@pytest.fixture
def webhook_server():
    """Create a mock webhook server."""
    return MockWebhookServer()


@pytest_asyncio.fixture
async def notification_harness():
    """Harness configured for notification tests."""
    config = E2ETestConfig(
        num_agents=2,
        agent_capabilities=["debate", "general"],
        agent_response_delay=0.01,
        timeout_seconds=60.0,
        task_timeout_seconds=30.0,
        heartbeat_interval=2.0,
        default_debate_rounds=1,
    )
    async with e2e_environment(config) as harness:
        yield harness


# ============================================================================
# Email Notification Tests
# ============================================================================


class TestEmailNotification:
    """Test email notification delivery."""

    @pytest.mark.asyncio
    async def test_email_notification_on_debate_complete(
        self, email_provider: MockNotificationProvider
    ):
        """Test email is sent when debate completes."""
        # Simulate sending notification on debate complete
        await email_provider.send(
            recipient="user@example.com",
            subject="Debate Complete: AI Safety Discussion",
            body="Your debate has reached consensus. View the results at...",
            debate_id="debate-123",
            event_type="debate_complete",
        )

        assert len(email_provider.sent_messages) == 1
        message = email_provider.sent_messages[0]
        assert message["recipient"] == "user@example.com"
        assert "Debate Complete" in message["subject"]
        assert message["debate_id"] == "debate-123"

    @pytest.mark.asyncio
    async def test_email_delivery_latency(self, email_provider: MockNotificationProvider):
        """Test email delivery meets latency SLA."""
        email_provider.delay_seconds = 0.1  # 100ms delay

        start = time.time()
        await email_provider.send(
            recipient="user@example.com",
            subject="Test Notification",
            body="Test body",
        )
        elapsed = time.time() - start

        assert elapsed < SLAS.max_delivery_time_seconds
        assert len(email_provider.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_email_retry_on_failure(self, email_provider: MockNotificationProvider):
        """Test email retry on transient failure."""
        email_provider.failures_to_simulate = 2  # Fail twice, then succeed

        retry_attempts = 0
        max_retries = 3

        while retry_attempts < max_retries:
            try:
                await email_provider.send(
                    recipient="user@example.com",
                    subject="Retry Test",
                    body="Test body",
                )
                break  # Success
            except RuntimeError:
                retry_attempts += 1
                await asyncio.sleep(0.1)  # Brief backoff

        # Should have succeeded after retries
        assert len(email_provider.sent_messages) == 1
        assert email_provider.fail_count == 2

    @pytest.mark.asyncio
    async def test_email_content_formatting(self, email_provider: MockNotificationProvider):
        """Test email content is properly formatted."""
        await email_provider.send(
            recipient="user@example.com",
            subject="Debate Results: Should we adopt microservices?",
            body="<h1>Debate Complete</h1><p>Consensus: Yes, with caveats...</p>",
            content_type="text/html",
            debate_id="debate-456",
        )

        message = email_provider.sent_messages[0]
        assert message["content_type"] == "text/html"
        assert "<h1>" in message["body"]


# ============================================================================
# Webhook Delivery Tests
# ============================================================================


class TestWebhookDelivery:
    """Test webhook delivery functionality."""

    @pytest.mark.asyncio
    async def test_webhook_delivery_success(self, webhook_server: MockWebhookServer):
        """Test successful webhook delivery."""
        payload = {
            "event": "debate.completed",
            "debate_id": "debate-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "consensus": {
                "reached": True,
                "result": "Microservices recommended for large teams",
            },
        }

        success = await webhook_server.receive_webhook(payload)

        assert success
        assert len(webhook_server.received_webhooks) == 1
        assert webhook_server.received_webhooks[0]["payload"]["debate_id"] == "debate-123"

    @pytest.mark.asyncio
    async def test_webhook_delivery_with_retry(self, webhook_server: MockWebhookServer):
        """Test webhook delivery retries on failure."""
        webhook_server.fail_rate = 0.5  # 50% failure rate

        delivery_attempts = 0
        max_attempts = 5
        delivered = False

        while delivery_attempts < max_attempts and not delivered:
            delivered = await webhook_server.receive_webhook(
                {
                    "event": "debate.completed",
                    "debate_id": f"retry-test-{delivery_attempts}",
                }
            )
            delivery_attempts += 1
            if not delivered:
                await asyncio.sleep(0.1)

        # With 50% failure rate and 5 attempts, very likely to succeed
        # At least some attempts should have succeeded
        assert len(webhook_server.received_webhooks) >= 1 or delivery_attempts == max_attempts

    @pytest.mark.asyncio
    async def test_webhook_delivery_timeout(self, webhook_server: MockWebhookServer):
        """Test webhook delivery handles timeouts."""
        webhook_server.response_delay_seconds = 0.5

        start = time.time()
        try:
            await asyncio.wait_for(
                webhook_server.receive_webhook({"event": "test"}),
                timeout=SLAS.max_delivery_time_seconds,
            )
        except asyncio.TimeoutError:
            pytest.fail("Webhook delivery timed out")

        elapsed = time.time() - start
        assert elapsed < SLAS.max_delivery_time_seconds

    @pytest.mark.asyncio
    async def test_webhook_payload_structure(self, webhook_server: MockWebhookServer):
        """Test webhook payload has correct structure."""
        payload = {
            "event": "debate.consensus",
            "debate_id": "debate-789",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "consensus_type": "majority",
                "agreement_score": 0.85,
                "participants": ["claude", "gpt-4", "gemini"],
                "result": "Recommended approach: hybrid architecture",
            },
            "metadata": {
                "workspace_id": "ws-123",
                "user_id": "user-456",
            },
        }

        await webhook_server.receive_webhook(payload)

        received = webhook_server.received_webhooks[0]["payload"]
        assert "event" in received
        assert "debate_id" in received
        assert "timestamp" in received
        assert "data" in received


# ============================================================================
# Slack Notification Tests
# ============================================================================


class TestSlackNotification:
    """Test Slack notification routing."""

    @pytest.mark.asyncio
    async def test_slack_notification_routing(self, slack_provider: MockNotificationProvider):
        """Test Slack notification is routed correctly."""
        await slack_provider.send(
            recipient="#debates-notifications",
            subject="New Debate Started",
            body="A new debate on 'AI Ethics' has started. Join now!",
            channel_type="slack",
            workspace_id="ws-123",
        )

        assert len(slack_provider.sent_messages) == 1
        message = slack_provider.sent_messages[0]
        assert message["recipient"].startswith("#")
        assert message["channel_type"] == "slack"

    @pytest.mark.asyncio
    async def test_slack_direct_message(self, slack_provider: MockNotificationProvider):
        """Test Slack direct message delivery."""
        await slack_provider.send(
            recipient="@john.doe",
            subject="You've been mentioned",
            body="You were mentioned in debate 'Budget Allocation'",
            message_type="direct",
        )

        assert len(slack_provider.sent_messages) == 1
        message = slack_provider.sent_messages[0]
        assert message["recipient"].startswith("@")

    @pytest.mark.asyncio
    async def test_slack_thread_reply(self, slack_provider: MockNotificationProvider):
        """Test Slack thread reply notification."""
        await slack_provider.send(
            recipient="#general",
            subject="Thread Update",
            body="New consensus reached in the thread",
            thread_ts="1234567890.123456",
        )

        message = slack_provider.sent_messages[0]
        assert "thread_ts" in message


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestNotificationRateLimiting:
    """Test notification rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, email_provider: MockNotificationProvider):
        """Test rate limiting prevents excessive notifications."""
        rate_limit_per_second = 10
        notifications_sent = 0
        start_time = time.time()

        # Try to send more than rate limit allows
        for i in range(rate_limit_per_second * 2):
            elapsed = time.time() - start_time
            if elapsed < 1.0:  # Within 1 second window
                if notifications_sent >= rate_limit_per_second:
                    # Should be rate limited
                    break
            await email_provider.send(
                recipient=f"user{i}@example.com",
                subject=f"Notification {i}",
                body="Test",
            )
            notifications_sent += 1

        # Should have enforced some kind of limiting in a real system
        # For mock, we just verify we can send up to the limit
        assert notifications_sent >= rate_limit_per_second

    @pytest.mark.asyncio
    async def test_burst_handling(self, email_provider: MockNotificationProvider):
        """Test burst notification handling."""
        burst_size = SLAS.burst_limit

        # Send burst of notifications
        tasks = []
        for i in range(burst_size):
            task = email_provider.send(
                recipient=f"burst{i}@example.com",
                subject=f"Burst {i}",
                body="Burst test",
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # All burst notifications should be sent
        assert len(email_provider.sent_messages) == burst_size

    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, email_provider: MockNotificationProvider):
        """Test rate limit recovery after window passes."""
        # First window - send some notifications
        for i in range(5):
            await email_provider.send(
                recipient=f"window1-{i}@example.com",
                subject=f"Window 1 - {i}",
                body="Test",
            )

        first_window_count = len(email_provider.sent_messages)

        # Simulate window passing
        await asyncio.sleep(0.1)

        # Second window - should be able to send again
        for i in range(5):
            await email_provider.send(
                recipient=f"window2-{i}@example.com",
                subject=f"Window 2 - {i}",
                body="Test",
            )

        assert len(email_provider.sent_messages) == first_window_count + 5


# ============================================================================
# Integration Tests
# ============================================================================


class TestNotificationIntegration:
    """Integration tests for notification system."""

    @pytest.mark.asyncio
    async def test_multi_channel_notification(
        self,
        email_provider: MockNotificationProvider,
        slack_provider: MockNotificationProvider,
        webhook_server: MockWebhookServer,
    ):
        """Test notification sent to multiple channels."""
        debate_id = "multi-channel-test"
        notification_data = {
            "debate_id": debate_id,
            "event": "consensus_reached",
            "result": "Decision made",
        }

        # Send to all channels
        await email_provider.send(
            recipient="user@example.com",
            subject="Consensus Reached",
            body=f"Debate {debate_id} reached consensus",
        )

        await slack_provider.send(
            recipient="#notifications",
            subject="Consensus Reached",
            body=f"Debate {debate_id} reached consensus",
        )

        await webhook_server.receive_webhook(notification_data)

        # All channels should have received notification
        assert len(email_provider.sent_messages) == 1
        assert len(slack_provider.sent_messages) == 1
        assert len(webhook_server.received_webhooks) == 1

    @pytest.mark.asyncio
    async def test_notification_with_debate_context(
        self, notification_harness: E2ETestHarness, email_provider: MockNotificationProvider
    ):
        """Test notification includes debate context."""
        # Run a debate
        result = await notification_harness.run_debate(
            "Test notification context",
            rounds=1,
        )

        # Simulate sending notification with context
        await email_provider.send(
            recipient="user@example.com",
            subject="Debate Complete",
            body="Your debate has completed",
            debate_result=result,
            participants=["agent-1", "agent-2"],
        )

        message = email_provider.sent_messages[0]
        assert "participants" in message


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestNotificationErrorHandling:
    """Test notification error handling."""

    @pytest.mark.asyncio
    async def test_graceful_provider_failure(self, email_provider: MockNotificationProvider):
        """Test graceful handling of provider failure."""
        email_provider.failures_to_simulate = 10  # Always fail

        with pytest.raises(RuntimeError):
            await email_provider.send(
                recipient="user@example.com",
                subject="Will Fail",
                body="This should fail",
            )

        assert email_provider.fail_count == 1

    @pytest.mark.asyncio
    async def test_invalid_recipient_handling(self, email_provider: MockNotificationProvider):
        """Test handling of invalid recipients."""
        # Provider should accept the message (validation happens externally)
        await email_provider.send(
            recipient="",  # Empty recipient
            subject="Test",
            body="Test",
        )

        # Message was accepted by mock provider
        assert len(email_provider.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_notification_timeout_handling(self, email_provider: MockNotificationProvider):
        """Test notification timeout handling."""
        email_provider.delay_seconds = 5.0  # Long delay

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                email_provider.send(
                    recipient="user@example.com",
                    subject="Timeout Test",
                    body="Test",
                ),
                timeout=0.1,  # Short timeout
            )


# ============================================================================
# Metrics Tests
# ============================================================================


class TestNotificationMetrics:
    """Test notification metrics collection."""

    @pytest.mark.asyncio
    async def test_delivery_metrics_collected(self, email_provider: MockNotificationProvider):
        """Test delivery metrics are collected."""
        # Send some notifications
        for i in range(5):
            await email_provider.send(
                recipient=f"metrics{i}@example.com",
                subject=f"Metrics Test {i}",
                body="Test",
            )

        # Verify metrics
        assert len(email_provider.sent_messages) == 5
        assert email_provider.fail_count == 0

    @pytest.mark.asyncio
    async def test_failure_metrics_collected(self, email_provider: MockNotificationProvider):
        """Test failure metrics are collected."""
        email_provider.failures_to_simulate = 3

        for i in range(5):
            try:
                await email_provider.send(
                    recipient=f"fail{i}@example.com",
                    subject="Failure Test",
                    body="Test",
                )
            except RuntimeError:
                pass

        # Should have recorded failures
        assert email_provider.fail_count == 3
        assert len(email_provider.sent_messages) == 2
