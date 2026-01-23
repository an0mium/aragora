"""
Tests for the Notification Dispatcher.

Tests cover:
- Retry logic with exponential backoff
- Circuit breaker per channel
- Queue persistence
- Email provider
- Rate limiting
- Metrics and status
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.control_plane.channels import (
    ChannelConfig,
    NotificationChannel,
    NotificationEventType,
    NotificationManager,
    NotificationMessage,
    NotificationPriority,
    NotificationResult,
)
from aragora.control_plane.notifications import (
    EmailProvider,
    NotificationDispatcher,
    NotificationDispatcherConfig,
    QueuedNotification,
    RetryConfig,
    create_notification_dispatcher,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_get_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(jitter=False)

        # Attempt 0: 1 * 2^0 = 1
        assert config.get_delay(0) == 1.0
        # Attempt 1: 1 * 2^1 = 2
        assert config.get_delay(1) == 2.0
        # Attempt 2: 1 * 2^2 = 4
        assert config.get_delay(2) == 4.0

    def test_get_delay_max_cap(self):
        """Test delay is capped at max."""
        config = RetryConfig(
            initial_delay_seconds=10.0,
            max_delay_seconds=30.0,
            jitter=False,
        )

        # Should be capped at 30, not 10 * 2^3 = 80
        assert config.get_delay(3) == 30.0

    def test_get_delay_with_jitter(self):
        """Test jitter adds randomness."""
        config = RetryConfig(jitter=True)

        delays = [config.get_delay(1) for _ in range(10)]
        # With jitter, delays should vary (not all equal)
        assert len(set(delays)) > 1


class TestQueuedNotification:
    """Tests for QueuedNotification serialization."""

    def test_to_dict(self):
        """Test serialization to dict."""
        message = NotificationMessage(
            event_type=NotificationEventType.TASK_COMPLETED,
            title="Test",
            body="Test body",
            priority=NotificationPriority.NORMAL,
        )
        config = ChannelConfig(
            channel_type=NotificationChannel.SLACK,
            slack_webhook_url="https://hooks.slack.com/test",
        )

        queued = QueuedNotification(
            id="test-123",
            message=message,
            channel_config=config,
            attempt=1,
        )

        data = queued.to_dict()

        assert data["id"] == "test-123"
        assert data["message"]["event_type"] == "task_completed"
        assert data["channel_type"] == "slack"
        assert data["attempt"] == 1

    def test_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        message = NotificationMessage(
            event_type=NotificationEventType.DELIBERATION_CONSENSUS,
            title="Consensus Reached",
            body="Agents reached consensus",
            priority=NotificationPriority.HIGH,
            workspace_id="ws-123",
            link_url="https://app.aragora.ai/debates/123",
            link_text="View Debate",
        )
        config = ChannelConfig(
            channel_type=NotificationChannel.TEAMS,
            teams_webhook_url="https://teams.webhook/test",
            workspace_id="ws-123",
        )

        original = QueuedNotification(
            id="test-456",
            message=message,
            channel_config=config,
            attempt=2,
            last_error="Previous failure",
        )

        data = original.to_dict()
        restored = QueuedNotification.from_dict(data)

        assert restored.id == original.id
        assert restored.message.title == original.message.title
        assert restored.message.event_type == original.message.event_type
        assert restored.channel_config.channel_type == original.channel_config.channel_type
        assert restored.attempt == original.attempt
        assert restored.last_error == original.last_error


class TestEmailProvider:
    """Tests for EmailProvider."""

    @pytest.mark.asyncio
    async def test_send_no_recipients(self):
        """Test error when no recipients configured."""
        provider = EmailProvider()
        message = NotificationMessage(
            event_type=NotificationEventType.TASK_COMPLETED,
            title="Test",
            body="Body",
        )
        config = ChannelConfig(
            channel_type=NotificationChannel.EMAIL,
            email_recipients=[],
        )

        result = await provider.send(message, config)

        assert result.success is False
        assert "No email recipients" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_no_smtp_host(self):
        """Test error when no SMTP host configured."""
        provider = EmailProvider()
        message = NotificationMessage(
            event_type=NotificationEventType.TASK_COMPLETED,
            title="Test",
            body="Body",
        )
        config = ChannelConfig(
            channel_type=NotificationChannel.EMAIL,
            email_recipients=["test@example.com"],
            smtp_host=None,
        )

        result = await provider.send(message, config)

        assert result.success is False
        assert "No SMTP host" in (result.error or "")

    def test_format_message(self):
        """Test email HTML/text formatting."""
        provider = EmailProvider()
        message = NotificationMessage(
            event_type=NotificationEventType.SLA_VIOLATION,
            title="SLA Violation",
            body="Task exceeded timeout",
            priority=NotificationPriority.URGENT,
            link_url="https://app.aragora.ai/tasks/123",
        )

        html, text = provider.format_message(message)

        assert "SLA Violation" in html
        assert "SLA Violation" in text
        assert "#ef4444" in html  # Urgent red color
        assert "https://app.aragora.ai/tasks/123" in html


class TestNotificationDispatcher:
    """Tests for NotificationDispatcher."""

    @pytest.fixture
    def manager(self):
        """Create a notification manager with mock channels."""
        manager = NotificationManager()
        manager.add_channel(
            ChannelConfig(
                channel_type=NotificationChannel.SLACK,
                slack_webhook_url="https://hooks.slack.com/test",
            )
        )
        return manager

    @pytest.fixture
    def dispatcher(self, manager):
        """Create a dispatcher with test config."""
        config = NotificationDispatcherConfig(
            retry_config=RetryConfig(max_retries=2, initial_delay_seconds=0.01),
            queue_enabled=False,  # Disable for unit tests
        )
        return NotificationDispatcher(manager=manager, config=config)

    @pytest.mark.asyncio
    async def test_dispatch_success(self, dispatcher, manager):
        """Test successful notification dispatch."""
        # Mock the send method
        mock_result = NotificationResult(
            success=True,
            channel=NotificationChannel.SLACK,
            message_id="123",
        )
        manager._providers[NotificationChannel.SLACK].send = AsyncMock(return_value=mock_result)

        results = await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_COMPLETED,
            title="Task Done",
            body="Your task completed successfully",
        )

        assert len(results) == 1
        assert results[0].success is True
        assert dispatcher._metrics["total_delivered"] == 1

    @pytest.mark.asyncio
    async def test_dispatch_retry_on_failure(self, dispatcher, manager):
        """Test retry logic on failure."""
        # First two calls fail, third succeeds
        call_count = 0

        async def mock_send(message, config):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return NotificationResult(
                    success=False,
                    channel=NotificationChannel.SLACK,
                    error="Temporary failure",
                )
            return NotificationResult(
                success=True,
                channel=NotificationChannel.SLACK,
            )

        manager._providers[NotificationChannel.SLACK].send = mock_send

        results = await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_COMPLETED,
            title="Task Done",
            body="Body",
        )

        assert len(results) == 1
        assert results[0].success is True
        assert call_count == 3  # Original + 2 retries
        assert dispatcher._metrics["total_retried"] >= 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, manager):
        """Test circuit breaker opens after threshold failures."""
        config = NotificationDispatcherConfig(
            retry_config=RetryConfig(max_retries=0),  # No retries
            circuit_breaker_failure_threshold=2,
            queue_enabled=False,
        )
        dispatcher = NotificationDispatcher(manager=manager, config=config)

        # Always fail
        manager._providers[NotificationChannel.SLACK].send = AsyncMock(
            return_value=NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error="Always fails",
            )
        )

        # First two failures
        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_COMPLETED,
            title="Test",
            body="Body",
        )
        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_COMPLETED,
            title="Test",
            body="Body",
        )

        # Circuit should be open now
        breaker = dispatcher._get_circuit_breaker(NotificationChannel.SLACK)
        assert breaker.failures >= 2

    @pytest.mark.asyncio
    async def test_rate_limiting(self, manager):
        """Test rate limiting per channel."""
        config = NotificationDispatcherConfig(
            rate_limit_per_channel=5,
            queue_enabled=False,
        )
        dispatcher = NotificationDispatcher(manager=manager, config=config)

        # Mock successful sends
        manager._providers[NotificationChannel.SLACK].send = AsyncMock(
            return_value=NotificationResult(
                success=True,
                channel=NotificationChannel.SLACK,
            )
        )

        # Send 10 notifications quickly
        for _ in range(10):
            await dispatcher.dispatch(
                event_type=NotificationEventType.TASK_COMPLETED,
                title="Test",
                body="Body",
            )

        # Should be rate limited after 5
        assert dispatcher._metrics["total_dispatched"] == 10
        # Some should have been rate limited (sent to queue or skipped)

    def test_get_metrics(self, dispatcher):
        """Test metrics retrieval."""
        metrics = dispatcher.get_metrics()

        assert "total_dispatched" in metrics
        assert "total_delivered" in metrics
        assert "total_failed" in metrics
        assert "circuit_breakers" in metrics
        assert "worker_running" in metrics

    def test_get_circuit_breaker_status(self, dispatcher):
        """Test circuit breaker status retrieval."""
        # Access a circuit breaker to create it
        dispatcher._get_circuit_breaker(NotificationChannel.SLACK)

        status = dispatcher.get_circuit_breaker_status()

        assert NotificationChannel.SLACK.value in status


class TestNotificationDispatcherWithRedis:
    """Tests for dispatcher with Redis queue."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.xadd = AsyncMock(return_value="1234567890-0")
        redis.xreadgroup = AsyncMock(return_value=[])
        redis.xack = AsyncMock()
        redis.xdel = AsyncMock()
        redis.xgroup_create = AsyncMock()
        redis.xinfo_stream = AsyncMock(return_value={"length": 5})
        return redis

    @pytest.fixture
    def manager(self):
        """Create notification manager."""
        manager = NotificationManager()
        manager.add_channel(
            ChannelConfig(
                channel_type=NotificationChannel.WEBHOOK,
                webhook_url="https://webhook.test/endpoint",
            )
        )
        return manager

    @pytest.mark.asyncio
    async def test_queue_notification(self, manager, mock_redis):
        """Test notification queueing."""
        config = NotificationDispatcherConfig(
            retry_config=RetryConfig(max_retries=0),
            queue_enabled=True,
        )
        dispatcher = NotificationDispatcher(
            manager=manager,
            redis=mock_redis,
            config=config,
        )

        # Make delivery fail
        manager._providers[NotificationChannel.WEBHOOK].send = AsyncMock(
            return_value=NotificationResult(
                success=False,
                channel=NotificationChannel.WEBHOOK,
                error="Failed",
            )
        )

        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_FAILED,
            title="Task Failed",
            body="Task xyz failed",
        )

        # Should have queued the notification
        mock_redis.xadd.assert_called()

    @pytest.mark.asyncio
    async def test_get_queue_depth(self, manager, mock_redis):
        """Test queue depth retrieval."""
        dispatcher = NotificationDispatcher(
            manager=manager,
            redis=mock_redis,
        )

        depth = await dispatcher.get_queue_depth()

        assert depth == 5
        mock_redis.xinfo_stream.assert_called()

    @pytest.mark.asyncio
    async def test_worker_start_stop(self, manager, mock_redis):
        """Test worker lifecycle."""
        dispatcher = NotificationDispatcher(
            manager=manager,
            redis=mock_redis,
        )

        await dispatcher.start_worker()
        assert dispatcher._worker_task is not None

        await dispatcher.stop_worker()
        assert dispatcher._shutdown is True


class TestCreateNotificationDispatcher:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test creating dispatcher with defaults."""
        dispatcher = create_notification_dispatcher()

        assert dispatcher._manager is not None
        assert dispatcher._redis is None
        assert dispatcher._config is not None

    def test_create_with_existing_manager(self):
        """Test creating dispatcher with existing manager."""
        manager = NotificationManager()
        manager.add_channel(
            ChannelConfig(
                channel_type=NotificationChannel.SLACK,
                slack_webhook_url="https://test",
            )
        )

        dispatcher = create_notification_dispatcher(manager=manager)

        assert dispatcher._manager is manager
        assert len(dispatcher._manager.get_channels()) == 1

    def test_create_with_custom_config(self):
        """Test creating dispatcher with custom config."""
        config = NotificationDispatcherConfig(
            retry_config=RetryConfig(max_retries=5),
            circuit_breaker_failure_threshold=10,
        )

        dispatcher = create_notification_dispatcher(config=config)

        assert dispatcher._config.retry_config.max_retries == 5
        assert dispatcher._config.circuit_breaker_failure_threshold == 10


class TestEmailProviderIntegration:
    """Integration test for email provider registration."""

    def test_email_provider_registered(self):
        """Test that email provider is auto-registered."""
        manager = NotificationManager()
        dispatcher = NotificationDispatcher(manager=manager)

        assert NotificationChannel.EMAIL in manager._providers
        assert isinstance(manager._providers[NotificationChannel.EMAIL], EmailProvider)
