"""
Tests for Integration Store Metrics.

Tests operation tracking, health monitoring, and metrics collection.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from aragora.storage.integration_store_metrics import (
    OperationMetrics,
    IntegrationStoreMetrics,
    InstrumentedIntegrationStore,
    get_metrics,
    reset_metrics,
    get_integration_metrics,
    get_integration_health,
)


class TestOperationMetrics:
    """Tests for OperationMetrics."""

    def test_initial_state(self):
        """Should have zero values initially."""
        metrics = OperationMetrics()
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.avg_latency_seconds == 0.0
        assert metrics.success_rate == 0.0

    def test_record_success(self):
        """Should track successful calls."""
        metrics = OperationMetrics()
        metrics.record_success(0.05)
        metrics.record_success(0.10)

        assert metrics.total_calls == 2
        assert metrics.successful_calls == 2
        assert metrics.failed_calls == 0
        assert abs(metrics.avg_latency_seconds - 0.075) < 0.001  # Float tolerance
        assert metrics.min_latency_seconds == 0.05
        assert metrics.max_latency_seconds == 0.10
        assert metrics.success_rate == 100.0
        assert metrics.last_call_at is not None

    def test_record_failure(self):
        """Should track failed calls."""
        metrics = OperationMetrics()
        metrics.record_success(0.05)
        metrics.record_failure("Connection error")

        assert metrics.total_calls == 2
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 1
        assert metrics.success_rate == 50.0
        assert metrics.last_error == "Connection error"
        assert metrics.last_error_at is not None

    def test_to_dict(self):
        """Should serialize to dictionary."""
        metrics = OperationMetrics()
        metrics.record_success(0.05)

        data = metrics.to_dict()
        assert "total_calls" in data
        assert "avg_latency_seconds" in data
        assert "success_rate" in data
        assert data["total_calls"] == 1


class TestIntegrationStoreMetrics:
    """Tests for IntegrationStoreMetrics."""

    def test_cache_tracking(self):
        """Should track cache hits and misses."""
        metrics = IntegrationStoreMetrics()

        metrics.record_cache_hit()
        metrics.record_cache_hit()
        metrics.record_cache_miss()

        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1
        assert round(metrics.cache_hit_rate, 2) == 66.67

    def test_to_dict_complete(self):
        """Should serialize all metrics."""
        metrics = IntegrationStoreMetrics()
        metrics.backend_type = "postgresql"
        metrics.is_healthy = True
        metrics.get_operations.record_success(0.01)
        metrics.save_operations.record_success(0.02)

        data = metrics.to_dict()

        assert data["backend_type"] == "postgresql"
        assert data["is_healthy"] is True
        assert "operations" in data
        assert "get" in data["operations"]
        assert "save" in data["operations"]
        assert data["operations"]["get"]["total_calls"] == 1
        assert data["operations"]["save"]["total_calls"] == 1


class TestInstrumentedIntegrationStore:
    """Tests for InstrumentedIntegrationStore wrapper."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    @pytest.fixture
    def mock_store(self):
        """Create a mock base store."""
        store = MagicMock()
        store.get_async = AsyncMock(return_value={"type": "test"})
        store.save_async = AsyncMock()
        store.delete_async = AsyncMock(return_value=True)
        store.list_for_user = AsyncMock(return_value=[{"type": "test1"}, {"type": "test2"}])
        store.list_all = AsyncMock(
            return_value=[{"type": "test1"}, {"type": "test2"}, {"type": "test3"}]
        )
        return store

    @pytest.mark.asyncio
    async def test_get_records_latency(self, mock_store):
        """Should record latency for get operations."""
        instrumented = InstrumentedIntegrationStore(mock_store, "postgresql")

        result = await instrumented.get("test", "user-1")

        assert result == {"type": "test"}
        metrics = instrumented.metrics
        assert metrics.get_operations.total_calls == 1
        assert metrics.get_operations.successful_calls == 1
        assert metrics.get_operations.avg_latency_seconds > 0

    @pytest.mark.asyncio
    async def test_save_records_latency(self, mock_store):
        """Should record latency for save operations."""
        instrumented = InstrumentedIntegrationStore(mock_store, "sqlite")

        await instrumented.save({"type": "test", "user_id": "user-1"})

        metrics = instrumented.metrics
        assert metrics.save_operations.total_calls == 1
        assert metrics.save_operations.successful_calls == 1

    @pytest.mark.asyncio
    async def test_delete_records_latency(self, mock_store):
        """Should record latency for delete operations."""
        instrumented = InstrumentedIntegrationStore(mock_store)

        result = await instrumented.delete("test", "user-1")

        assert result is True
        metrics = instrumented.metrics
        assert metrics.delete_operations.total_calls == 1

    @pytest.mark.asyncio
    async def test_list_updates_active_count(self, mock_store):
        """Should update active integrations count on list."""
        instrumented = InstrumentedIntegrationStore(mock_store)

        result = await instrumented.list_for_user("user-1")

        assert len(result) == 2
        metrics = instrumented.metrics
        assert metrics.active_integrations == 2
        assert metrics.list_operations.total_calls == 1

    @pytest.mark.asyncio
    async def test_error_tracking(self, mock_store):
        """Should track errors and mark unhealthy after failures."""
        mock_store.get_async = AsyncMock(side_effect=RuntimeError("DB error"))
        instrumented = InstrumentedIntegrationStore(mock_store)

        # First two failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await instrumented.get("test")

        # Should still be healthy
        assert instrumented.metrics.is_healthy is True
        assert instrumented.metrics.consecutive_failures == 2

        # Third failure marks unhealthy
        with pytest.raises(RuntimeError):
            await instrumented.get("test")

        assert instrumented.metrics.is_healthy is False
        assert instrumented.metrics.consecutive_failures == 3
        assert instrumented.metrics.get_operations.last_error == "DB error"

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, mock_store):
        """Should reset failure count on success."""
        instrumented = InstrumentedIntegrationStore(mock_store)
        instrumented.metrics.consecutive_failures = 2
        instrumented.metrics.is_healthy = False

        # Successful call should reset
        await instrumented.get("test")

        assert instrumented.metrics.consecutive_failures == 0
        assert instrumented.metrics.is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check(self, mock_store):
        """Should perform health check."""
        instrumented = InstrumentedIntegrationStore(mock_store, "postgresql")

        health = await instrumented.health_check()

        assert health["healthy"] is True
        assert health["backend_type"] == "postgresql"
        assert health["consecutive_failures"] == 0
        assert "last_check" in health

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_store):
        """Should mark unhealthy on health check failure."""
        mock_store.list_all = AsyncMock(side_effect=RuntimeError("Connection refused"))
        instrumented = InstrumentedIntegrationStore(mock_store)

        health = await instrumented.health_check()

        assert health["healthy"] is False


class TestGlobalMetricsFunctions:
    """Tests for global metrics functions."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    @pytest.mark.asyncio
    async def test_get_integration_metrics(self):
        """Should return complete metrics dict."""
        metrics = get_metrics()
        metrics.backend_type = "redis"
        metrics.get_operations.record_success(0.01)
        metrics.save_operations.record_failure("Write error")

        result = await get_integration_metrics()

        assert result["backend_type"] == "redis"
        assert result["operations"]["get"]["total_calls"] == 1
        assert result["operations"]["save"]["failed_calls"] == 1

    @pytest.mark.asyncio
    async def test_get_integration_health(self):
        """Should return health status."""
        metrics = get_metrics()
        metrics.is_healthy = True
        metrics.backend_type = "postgresql"
        metrics.get_operations.record_success(0.01)

        result = await get_integration_health()

        assert result["healthy"] is True
        assert result["backend_type"] == "postgresql"
        assert "operations_summary" in result

    def test_reset_metrics(self):
        """Should reset global metrics."""
        metrics = get_metrics()
        metrics.get_operations.record_success(0.01)
        assert metrics.get_operations.total_calls == 1

        reset_metrics()

        new_metrics = get_metrics()
        assert new_metrics.get_operations.total_calls == 0


class TestMultipleOperations:
    """Tests for multiple operations tracking."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Should track concurrent operations correctly."""
        mock_store = MagicMock()

        async def slow_get(*args):
            await asyncio.sleep(0.01)
            return {"type": "test"}

        mock_store.get_async = slow_get
        mock_store.list_all = AsyncMock(return_value=[])

        instrumented = InstrumentedIntegrationStore(mock_store)

        # Run multiple concurrent gets
        await asyncio.gather(
            instrumented.get("test1"),
            instrumented.get("test2"),
            instrumented.get("test3"),
        )

        metrics = instrumented.metrics
        assert metrics.get_operations.total_calls == 3
        assert metrics.get_operations.successful_calls == 3

    @pytest.mark.asyncio
    async def test_mixed_operations(self):
        """Should track different operation types separately."""
        mock_store = MagicMock()
        mock_store.get_async = AsyncMock(return_value={"type": "test"})
        mock_store.save_async = AsyncMock()
        mock_store.list_all = AsyncMock(return_value=[{"type": "test"}])

        instrumented = InstrumentedIntegrationStore(mock_store)

        await instrumented.get("test")
        await instrumented.get("test")
        await instrumented.save({"type": "test"})
        await instrumented.list_all()

        metrics = instrumented.metrics
        assert metrics.get_operations.total_calls == 2
        assert metrics.save_operations.total_calls == 1
        assert metrics.list_operations.total_calls == 1
