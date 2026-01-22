"""
Tests for PostgreSQL connection pool resilience features.

Tests circuit breaker, timeouts, backpressure, and metrics for pool exhaustion handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


class TestPoolMetrics:
    """Tests for pool metrics collection."""

    def test_pool_metrics_returns_none_when_no_pool(self):
        """Metrics should return None when pool not initialized."""
        from aragora.storage.postgres_store import get_pool_metrics, _pool

        # Save and clear pool
        import aragora.storage.postgres_store as module

        original_pool = module._pool
        module._pool = None

        try:
            metrics = get_pool_metrics()
            assert metrics is None
        finally:
            module._pool = original_pool

    def test_pool_metrics_calculates_utilization(self):
        """Metrics should calculate pool utilization correctly."""
        from aragora.storage.postgres_store import get_pool_metrics, PoolMetrics
        import aragora.storage.postgres_store as module

        # Create mock pool
        mock_pool = MagicMock()
        mock_pool.get_size.return_value = 15
        mock_pool.get_min_size.return_value = 5
        mock_pool.get_max_size.return_value = 20
        mock_pool.get_idle_size.return_value = 5  # 10 used

        original_pool = module._pool
        module._pool = mock_pool

        try:
            metrics = get_pool_metrics()
            assert metrics is not None
            assert metrics.pool_size == 15
            assert metrics.pool_min_size == 5
            assert metrics.pool_max_size == 20
            assert metrics.free_connections == 5
            assert metrics.used_connections == 10
            assert metrics.utilization == 0.5  # 10/20
            assert metrics.backpressure is False  # Below 80%
        finally:
            module._pool = original_pool

    def test_backpressure_triggers_at_threshold(self):
        """Backpressure should trigger when utilization >= 80%."""
        from aragora.storage.postgres_store import get_pool_metrics
        import aragora.storage.postgres_store as module

        mock_pool = MagicMock()
        mock_pool.get_size.return_value = 20
        mock_pool.get_min_size.return_value = 5
        mock_pool.get_max_size.return_value = 20
        mock_pool.get_idle_size.return_value = 2  # 18 used = 90%

        original_pool = module._pool
        module._pool = mock_pool

        try:
            metrics = get_pool_metrics()
            assert metrics is not None
            assert metrics.utilization == 0.9
            assert metrics.backpressure is True
        finally:
            module._pool = original_pool

    def test_reset_pool_metrics(self):
        """Reset should clear all metrics."""
        from aragora.storage.postgres_store import reset_pool_metrics, _pool_metrics
        import aragora.storage.postgres_store as module

        # Set some metrics
        module._pool_metrics["total_acquisitions"] = 100
        module._pool_metrics["failed_acquisitions"] = 10
        module._pool_metrics["timeouts"] = 5

        reset_pool_metrics()

        assert module._pool_metrics["total_acquisitions"] == 0
        assert module._pool_metrics["failed_acquisitions"] == 0
        assert module._pool_metrics["timeouts"] == 0


class TestIsPoolHealthy:
    """Tests for pool health check."""

    def test_healthy_when_low_utilization(self):
        """Pool should be healthy when utilization is low."""
        from aragora.storage.postgres_store import is_pool_healthy
        import aragora.storage.postgres_store as module

        mock_pool = MagicMock()
        mock_pool.get_size.return_value = 10
        mock_pool.get_min_size.return_value = 5
        mock_pool.get_max_size.return_value = 20
        mock_pool.get_idle_size.return_value = 8  # 2 used = 10%

        original_pool = module._pool
        module._pool = mock_pool

        try:
            assert is_pool_healthy() is True
        finally:
            module._pool = original_pool

    def test_unhealthy_when_no_pool(self):
        """Pool should be unhealthy when not initialized."""
        from aragora.storage.postgres_store import is_pool_healthy
        import aragora.storage.postgres_store as module

        original_pool = module._pool
        module._pool = None

        try:
            assert is_pool_healthy() is False
        finally:
            module._pool = original_pool


class TestPoolExhaustedError:
    """Tests for PoolExhaustedError exception."""

    def test_error_message_format(self):
        """Error message should include timeout and utilization."""
        from aragora.storage.postgres_store import PoolExhaustedError

        error = PoolExhaustedError(timeout=10.0, utilization=0.95)
        assert "10.0s" in str(error)
        assert "95" in str(error) or "0.95" in str(error)

    def test_error_attributes(self):
        """Error should store timeout and utilization."""
        from aragora.storage.postgres_store import PoolExhaustedError

        error = PoolExhaustedError(timeout=5.0, utilization=0.8)
        assert error.timeout == 5.0
        assert error.utilization == 0.8


import sys

@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout requires Python 3.11+")
class TestAcquireConnectionResilient:
    """Tests for resilient connection acquisition."""

    @pytest.mark.asyncio
    async def test_successful_acquisition_records_metrics(self):
        """Successful acquisition should update metrics."""
        from aragora.storage.postgres_store import (
            acquire_connection_resilient,
            reset_pool_metrics,
        )
        import aragora.storage.postgres_store as module

        reset_pool_metrics()

        mock_conn = AsyncMock()
        mock_pool = MagicMock()

        # Create async context manager for acquire
        async def mock_acquire():
            return mock_conn

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.get_size.return_value = 10
        mock_pool.get_max_size.return_value = 20

        async with acquire_connection_resilient(mock_pool, timeout=5.0) as conn:
            assert conn == mock_conn

        assert module._pool_metrics["total_acquisitions"] == 1
        assert module._pool_metrics["failed_acquisitions"] == 0

    @pytest.mark.asyncio
    async def test_timeout_increments_metrics(self):
        """Timeout should increment timeout and failure metrics."""
        from aragora.storage.postgres_store import (
            acquire_connection_resilient,
            reset_pool_metrics,
            PoolExhaustedError,
        )
        import aragora.storage.postgres_store as module

        reset_pool_metrics()

        mock_pool = MagicMock()
        mock_pool.get_size.return_value = 20
        mock_pool.get_max_size.return_value = 20

        # Create a proper async context manager that times out
        class SlowAcquire:
            async def __aenter__(self):
                await asyncio.sleep(10)
                return MagicMock()

            async def __aexit__(self, *args):
                pass

        mock_pool.acquire.return_value = SlowAcquire()

        with pytest.raises(PoolExhaustedError):
            async with acquire_connection_resilient(
                mock_pool, timeout=0.1, retries=1, backoff_base=0.01
            ):
                pass

        assert module._pool_metrics["timeouts"] >= 1
        assert module._pool_metrics["failed_acquisitions"] >= 1


class TestPostgresStoreResilient:
    """Tests for PostgresStore with resilient connection handling."""

    def test_init_with_resilient_defaults(self):
        """Store should enable resilient mode by default."""
        import aragora.storage.postgres_store as module
        from aragora.storage.postgres_store import PostgresStore, POOL_ACQUIRE_TIMEOUT

        # Mock asyncpg availability
        original_available = module.ASYNCPG_AVAILABLE
        module.ASYNCPG_AVAILABLE = True

        try:
            mock_pool = MagicMock()

            class TestStore(PostgresStore):
                SCHEMA_NAME = "test_store"
                SCHEMA_VERSION = 1
                INITIAL_SCHEMA = "CREATE TABLE IF NOT EXISTS test (id TEXT PRIMARY KEY);"

            store = TestStore(mock_pool)
            assert store._use_resilient is True
            assert store._acquire_timeout == POOL_ACQUIRE_TIMEOUT
            assert store._acquire_retries == 3
        finally:
            module.ASYNCPG_AVAILABLE = original_available

    def test_init_with_custom_resilience_settings(self):
        """Store should accept custom resilience settings."""
        import aragora.storage.postgres_store as module
        from aragora.storage.postgres_store import PostgresStore

        # Mock asyncpg availability
        original_available = module.ASYNCPG_AVAILABLE
        module.ASYNCPG_AVAILABLE = True

        try:
            mock_pool = MagicMock()

            class TestStore(PostgresStore):
                SCHEMA_NAME = "test_store"
                SCHEMA_VERSION = 1
                INITIAL_SCHEMA = "CREATE TABLE IF NOT EXISTS test (id TEXT PRIMARY KEY);"

            store = TestStore(
                mock_pool,
                use_resilient=False,
                acquire_timeout=30.0,
                acquire_retries=5,
            )
            assert store._use_resilient is False
            assert store._acquire_timeout == 30.0
            assert store._acquire_retries == 5
        finally:
            module.ASYNCPG_AVAILABLE = original_available


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with pool acquisition."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Circuit breaker should open after repeated failures."""
        from aragora.storage.postgres_store import (
            acquire_connection_resilient,
            reset_pool_metrics,
            POOL_CIRCUIT_BREAKER_THRESHOLD,
        )
        from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers
        import aragora.storage.postgres_store as module

        reset_pool_metrics()
        reset_all_circuit_breakers()

        mock_pool = MagicMock()
        mock_pool.get_size.return_value = 20
        mock_pool.get_max_size.return_value = 20

        # Make acquire always timeout
        async def slow_acquire():
            await asyncio.sleep(10)

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = slow_acquire
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # Exhaust retries multiple times to trigger circuit breaker
        for _ in range(POOL_CIRCUIT_BREAKER_THRESHOLD + 1):
            try:
                async with acquire_connection_resilient(
                    mock_pool, timeout=0.05, retries=1, backoff_base=0.01
                ):
                    pass
            except Exception:
                pass

        # Check circuit breaker is open
        cb = get_circuit_breaker("postgres_pool")
        assert cb.get_status() in ("open", "half-open")


class TestExportedSymbols:
    """Tests for module exports."""

    def test_all_new_exports_available(self):
        """All new exports should be importable."""
        from aragora.storage.postgres_store import (
            acquire_connection_resilient,
            get_pool_metrics,
            is_pool_healthy,
            reset_pool_metrics,
            PoolMetrics,
            PoolExhaustedError,
            POOL_ACQUIRE_TIMEOUT,
            POOL_BACKPRESSURE_THRESHOLD,
            POOL_CIRCUIT_BREAKER_THRESHOLD,
            POOL_CIRCUIT_BREAKER_COOLDOWN,
        )

        # Just verify imports work
        assert callable(acquire_connection_resilient)
        assert callable(get_pool_metrics)
        assert callable(is_pool_healthy)
        assert callable(reset_pool_metrics)
        assert PoolMetrics is not None
        assert issubclass(PoolExhaustedError, Exception)
        assert POOL_ACQUIRE_TIMEOUT > 0
        assert 0 < POOL_BACKPRESSURE_THRESHOLD < 1
        assert POOL_CIRCUIT_BREAKER_THRESHOLD > 0
        assert POOL_CIRCUIT_BREAKER_COOLDOWN > 0
