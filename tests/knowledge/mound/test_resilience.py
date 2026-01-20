"""
Tests for Knowledge Mound Persistence Resilience.

Tests cover:
- Retry logic with exponential backoff
- Transaction management
- Cache invalidation events
- Health monitoring
- Integrity verification
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_exponential_backoff_delay(self):
        """Should calculate exponential delays."""
        from aragora.knowledge.mound.resilience import RetryConfig, RetryStrategy

        config = RetryConfig(
            base_delay=0.1,
            max_delay=10.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
        )

        assert config.calculate_delay(0) == 0.1  # 0.1 * 2^0
        assert config.calculate_delay(1) == 0.2  # 0.1 * 2^1
        assert config.calculate_delay(2) == 0.4  # 0.1 * 2^2
        assert config.calculate_delay(3) == 0.8  # 0.1 * 2^3

    def test_max_delay_cap(self):
        """Should cap delay at max_delay."""
        from aragora.knowledge.mound.resilience import RetryConfig, RetryStrategy

        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
        )

        # 2^10 * 1.0 = 1024, but should be capped at 5.0
        assert config.calculate_delay(10) == 5.0

    def test_linear_backoff_delay(self):
        """Should calculate linear delays."""
        from aragora.knowledge.mound.resilience import RetryConfig, RetryStrategy

        config = RetryConfig(
            base_delay=0.5,
            strategy=RetryStrategy.LINEAR,
            jitter=False,
        )

        assert config.calculate_delay(0) == 0.5  # 0.5 * 1
        assert config.calculate_delay(1) == 1.0  # 0.5 * 2
        assert config.calculate_delay(2) == 1.5  # 0.5 * 3

    def test_constant_delay(self):
        """Should return constant delay."""
        from aragora.knowledge.mound.resilience import RetryConfig, RetryStrategy

        config = RetryConfig(
            base_delay=0.25,
            strategy=RetryStrategy.CONSTANT,
            jitter=False,
        )

        assert config.calculate_delay(0) == 0.25
        assert config.calculate_delay(5) == 0.25
        assert config.calculate_delay(10) == 0.25

    def test_jitter_adds_variance(self):
        """Should add jitter to delay."""
        from aragora.knowledge.mound.resilience import RetryConfig, RetryStrategy

        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.CONSTANT,
            jitter=True,
        )

        delays = [config.calculate_delay(0) for _ in range(10)]
        # With jitter, delays should vary (±25%)
        assert min(delays) >= 0.75
        assert max(delays) <= 1.25
        # They shouldn't all be the same
        assert len(set(delays)) > 1


class TestWithRetry:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self):
        """Should not retry on success."""
        from aragora.knowledge.mound.resilience import with_retry, RetryConfig

        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Should retry on ConnectionError."""
        from aragora.knowledge.mound.resilience import with_retry, RetryConfig

        call_count = 0

        @with_retry(RetryConfig(max_retries=3, base_delay=0.01))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Should raise after max retries."""
        from aragora.knowledge.mound.resilience import with_retry, RetryConfig

        call_count = 0

        @with_retry(RetryConfig(max_retries=2, base_delay=0.01))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await always_fails()

        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable(self):
        """Should not retry non-retryable exceptions."""
        from aragora.knowledge.mound.resilience import with_retry, RetryConfig

        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await raises_value_error()

        assert call_count == 1  # No retries


class TestCacheInvalidationBus:
    """Tests for CacheInvalidationBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Should deliver events to subscribers."""
        from aragora.knowledge.mound.resilience import (
            CacheInvalidationBus,
            CacheInvalidationEvent,
        )

        bus = CacheInvalidationBus()
        received_events = []

        async def handler(event: CacheInvalidationEvent):
            received_events.append(event)

        bus.subscribe(handler)

        await bus.publish(
            CacheInvalidationEvent(
                event_type="node_updated",
                workspace_id="ws-1",
                item_id="node-1",
            )
        )

        assert len(received_events) == 1
        assert received_events[0].event_type == "node_updated"
        assert received_events[0].workspace_id == "ws-1"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Should stop delivering after unsubscribe."""
        from aragora.knowledge.mound.resilience import (
            CacheInvalidationBus,
            CacheInvalidationEvent,
        )

        bus = CacheInvalidationBus()
        received_events = []

        async def handler(event: CacheInvalidationEvent):
            received_events.append(event)

        unsubscribe = bus.subscribe(handler)

        await bus.publish_node_update("ws-1", "node-1")
        assert len(received_events) == 1

        unsubscribe()

        await bus.publish_node_update("ws-1", "node-2")
        assert len(received_events) == 1  # Still 1, no new events

    @pytest.mark.asyncio
    async def test_publish_node_update(self):
        """Should create node_updated event."""
        from aragora.knowledge.mound.resilience import CacheInvalidationBus

        bus = CacheInvalidationBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(handler)
        await bus.publish_node_update("ws-1", "node-1", foo="bar")

        assert received[0].event_type == "node_updated"
        assert received[0].metadata["foo"] == "bar"

    @pytest.mark.asyncio
    async def test_publish_node_delete(self):
        """Should create node_deleted event."""
        from aragora.knowledge.mound.resilience import CacheInvalidationBus

        bus = CacheInvalidationBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(handler)
        await bus.publish_node_delete("ws-1", "node-1")

        assert received[0].event_type == "node_deleted"

    @pytest.mark.asyncio
    async def test_subscriber_error_handled(self):
        """Should continue despite subscriber errors."""
        from aragora.knowledge.mound.resilience import CacheInvalidationBus

        bus = CacheInvalidationBus()
        successful_events = []

        async def failing_handler(event):
            raise ValueError("Handler error")

        async def success_handler(event):
            successful_events.append(event)

        bus.subscribe(failing_handler)
        bus.subscribe(success_handler)

        # Should not raise, and second handler should still receive
        await bus.publish_node_update("ws-1", "node-1")
        assert len(successful_events) == 1

    def test_get_recent_events(self):
        """Should return recent events."""
        from aragora.knowledge.mound.resilience import (
            CacheInvalidationBus,
            CacheInvalidationEvent,
        )

        bus = CacheInvalidationBus()
        for i in range(5):
            bus._event_log.append(
                CacheInvalidationEvent(
                    event_type="test",
                    workspace_id=f"ws-{i}",
                )
            )

        events = bus.get_recent_events(limit=3)
        assert len(events) == 3
        assert events[-1]["workspace_id"] == "ws-4"


class TestGlobalInvalidationBus:
    """Tests for global invalidation bus singleton."""

    def test_get_invalidation_bus_singleton(self):
        """Should return same instance."""
        from aragora.knowledge.mound.resilience import get_invalidation_bus

        bus1 = get_invalidation_bus()
        bus2 = get_invalidation_bus()

        assert bus1 is bus2


class TestHealthStatus:
    """Tests for HealthStatus."""

    def test_to_dict(self):
        """Should serialize to dict."""
        from aragora.knowledge.mound.resilience import HealthStatus

        status = HealthStatus(
            healthy=True,
            last_check=datetime(2024, 1, 1, tzinfo=timezone.utc),
            consecutive_failures=0,
            latency_ms=5.5,
        )

        data = status.to_dict()
        assert data["healthy"] is True
        assert data["latency_ms"] == 5.5
        assert "2024-01-01" in data["last_check"]


class TestIntegrityCheckResult:
    """Tests for IntegrityCheckResult."""

    def test_to_dict(self):
        """Should serialize to dict."""
        from aragora.knowledge.mound.resilience import IntegrityCheckResult

        result = IntegrityCheckResult(
            passed=False,
            checks_performed=5,
            issues_found=["Orphaned records"],
            details={"orphans": 3},
        )

        data = result.to_dict()
        assert data["passed"] is False
        assert data["checks_performed"] == 5
        assert "Orphaned records" in data["issues_found"]


class TestCacheInvalidationEvent:
    """Tests for CacheInvalidationEvent."""

    def test_to_dict(self):
        """Should serialize to dict."""
        from aragora.knowledge.mound.resilience import CacheInvalidationEvent

        event = CacheInvalidationEvent(
            event_type="node_updated",
            workspace_id="ws-1",
            item_id="node-1",
            metadata={"source": "test"},
        )

        data = event.to_dict()
        assert data["event_type"] == "node_updated"
        assert data["workspace_id"] == "ws-1"
        assert data["item_id"] == "node-1"
        assert data["metadata"]["source"] == "test"


class TestTransactionIsolation:
    """Tests for TransactionIsolation enum."""

    def test_isolation_levels(self):
        """Should have standard isolation levels."""
        from aragora.knowledge.mound.resilience import TransactionIsolation

        assert TransactionIsolation.READ_COMMITTED.value == "READ COMMITTED"
        assert TransactionIsolation.REPEATABLE_READ.value == "REPEATABLE READ"
        assert TransactionIsolation.SERIALIZABLE.value == "SERIALIZABLE"


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_strategies(self):
        """Should have expected strategies."""
        from aragora.knowledge.mound.resilience import RetryStrategy

        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.CONSTANT.value == "constant"


class TestTransactionConfig:
    """Tests for TransactionConfig."""

    def test_defaults(self):
        """Should have sensible defaults."""
        from aragora.knowledge.mound.resilience import (
            TransactionConfig,
            TransactionIsolation,
        )

        config = TransactionConfig()
        assert config.isolation == TransactionIsolation.READ_COMMITTED
        assert config.timeout_seconds == 30.0
        assert config.savepoint_on_nested is True


class TestConnectionHealthMonitor:
    """Tests for ConnectionHealthMonitor."""

    def test_record_success(self):
        """Should reset failure count on success."""
        from aragora.knowledge.mound.resilience import ConnectionHealthMonitor

        mock_pool = MagicMock()
        monitor = ConnectionHealthMonitor(mock_pool)
        monitor._status.consecutive_failures = 3
        monitor._status.healthy = False

        monitor.record_success()

        assert monitor._status.consecutive_failures == 0
        assert monitor._status.healthy is True

    def test_record_failure(self):
        """Should increment failure count."""
        from aragora.knowledge.mound.resilience import ConnectionHealthMonitor

        mock_pool = MagicMock()
        monitor = ConnectionHealthMonitor(mock_pool, failure_threshold=3)

        monitor.record_failure("Connection timeout")
        assert monitor._status.consecutive_failures == 1
        assert monitor._status.healthy is True

        monitor.record_failure("Connection timeout")
        monitor.record_failure("Connection timeout")
        assert monitor._status.consecutive_failures == 3
        assert monitor._status.healthy is False

    def test_is_healthy(self):
        """Should report health status."""
        from aragora.knowledge.mound.resilience import ConnectionHealthMonitor

        mock_pool = MagicMock()
        monitor = ConnectionHealthMonitor(mock_pool)

        assert monitor.is_healthy() is True

        monitor._status.healthy = False
        assert monitor.is_healthy() is False

    def test_get_status(self):
        """Should return current status."""
        from aragora.knowledge.mound.resilience import ConnectionHealthMonitor

        mock_pool = MagicMock()
        monitor = ConnectionHealthMonitor(mock_pool)

        status = monitor.get_status()
        assert status.healthy is True


class TestTransactionManager:
    """Tests for TransactionManager."""

    @pytest.mark.asyncio
    async def test_transaction_stats(self):
        """Should track transaction statistics."""
        from aragora.knowledge.mound.resilience import TransactionManager

        mock_pool = MagicMock()
        manager = TransactionManager(mock_pool)

        stats = manager.get_stats()
        assert stats["active_transactions"] == 0
        assert "default_isolation" in stats
        assert "default_timeout" in stats


class TestIntegrityVerifier:
    """Tests for IntegrityVerifier (mock-based)."""

    @pytest.mark.asyncio
    async def test_verify_all_structure(self):
        """Should return IntegrityCheckResult."""
        from aragora.knowledge.mound.resilience import IntegrityVerifier

        # Create mock pool with async context manager support
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=0)
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        verifier = IntegrityVerifier(mock_pool)
        result = await verifier.verify_all()

        assert result.passed is True
        assert result.checks_performed > 0
        assert isinstance(result.issues_found, list)
        assert isinstance(result.details, dict)

    @pytest.mark.asyncio
    async def test_verify_all_finds_issues(self):
        """Should report issues when found."""
        from aragora.knowledge.mound.resilience import IntegrityVerifier

        mock_conn = AsyncMock()
        # Return non-zero orphan counts
        mock_conn.fetchval = AsyncMock(side_effect=[5, 3, 2, 1, 0])
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        verifier = IntegrityVerifier(mock_pool)
        result = await verifier.verify_all()

        assert result.passed is False
        assert len(result.issues_found) > 0
        assert any("orphaned" in issue.lower() for issue in result.issues_found)


class TestResilientPostgresStore:
    """Tests for ResilientPostgresStore."""

    def test_get_health_status_not_initialized(self):
        """Should return basic status when not initialized."""
        from aragora.knowledge.mound.resilience import ResilientPostgresStore

        mock_store = MagicMock()
        mock_store._initialized = False

        resilient = ResilientPostgresStore(mock_store)
        status = resilient.get_health_status()

        assert status["initialized"] is False

    def test_is_healthy_no_monitor(self):
        """Should return True when no monitor configured."""
        from aragora.knowledge.mound.resilience import ResilientPostgresStore

        mock_store = MagicMock()
        resilient = ResilientPostgresStore(mock_store)

        assert resilient.is_healthy() is True
