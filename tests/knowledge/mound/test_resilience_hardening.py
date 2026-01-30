"""
Tests for Knowledge Mound resilience hardening.

Tests cover:
- Batch timeout enforcement (per-item and total)
- Deadlock-aware transaction retry
- Circuit breaker integration with coordinator
- Per-adapter circuit breaker configuration
- Health-aware circuit breaker
- Latency tracker and adaptive timeouts
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Batch Timeout Tests
# =============================================================================


class TestBatchTimeouts:
    """Tests for per-item and total batch timeout enforcement."""

    def _make_adapter(self, records: dict[str, Any] | None = None):
        """Create a test adapter with ReverseFlowMixin."""
        from aragora.knowledge.mound.adapters._reverse_flow_base import (
            ReverseFlowMixin,
        )

        class TestAdapter(ReverseFlowMixin):
            adapter_name = "test"

            def __init__(self, records: dict[str, Any]):
                self._records = records or {}

            def _get_record_for_validation(self, source_id: str) -> Any | None:
                return self._records.get(source_id)

            def _apply_km_validation(
                self,
                record: Any,
                km_confidence: float,
                cross_refs: Optional[list[str]] = None,
                metadata: Optional[dict[str, Any]] = None,
            ) -> bool:
                record["validated"] = True
                return True

        return TestAdapter(records or {})

    @pytest.mark.asyncio
    async def test_per_item_timeout_enforced(self):
        """Should timeout individual items that take too long."""
        from aragora.knowledge.mound.adapters._reverse_flow_base import (
            BatchTimeoutConfig,
        )

        adapter = self._make_adapter({"item1": {"id": "item1"}})

        # Monkey-patch _process_single_item to be slow
        original_process = adapter._process_single_item

        async def slow_process(*args, **kwargs):
            await asyncio.sleep(5)  # Simulate slow processing
            return await original_process(*args, **kwargs)

        adapter._process_single_item = slow_process

        items = [{"id": "item1", "confidence": 0.9, "metadata": {"source_id": "item1"}}]
        tc = BatchTimeoutConfig(per_item_timeout_seconds=0.05)

        result = await adapter.sync_validations_from_km(items, timeout_config=tc)

        assert result["records_analyzed"] == 1
        assert result["records_updated"] == 0
        assert len(result["errors"]) >= 1
        assert "Per-item timeout" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_batch_continues_on_timeout(self):
        """Should continue processing after a per-item timeout by default."""
        from aragora.knowledge.mound.adapters._reverse_flow_base import (
            BatchTimeoutConfig,
        )

        records = {"item1": {"id": "item1"}, "item2": {"id": "item2"}}
        adapter = self._make_adapter(records)

        call_count = 0
        original_process = adapter._process_single_item

        async def sometimes_slow(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(5)  # First item is slow
            return await original_process(*args, **kwargs)

        adapter._process_single_item = sometimes_slow

        items = [
            {"id": "item1", "confidence": 0.9, "metadata": {"source_id": "item1"}},
            {"id": "item2", "confidence": 0.9, "metadata": {"source_id": "item2"}},
        ]
        tc = BatchTimeoutConfig(
            per_item_timeout_seconds=0.05,
            fail_fast_on_timeout=False,
        )

        result = await adapter.sync_validations_from_km(items, timeout_config=tc)

        assert result["records_analyzed"] == 2
        assert result["records_updated"] == 1  # Second item succeeded
        assert len(result["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_fail_fast_stops_batch(self):
        """Should stop batch on first timeout when fail_fast=True."""
        from aragora.knowledge.mound.adapters._reverse_flow_base import (
            BatchTimeoutConfig,
        )

        records = {"item1": {"id": "item1"}, "item2": {"id": "item2"}}
        adapter = self._make_adapter(records)

        original_process = adapter._process_single_item

        async def slow_process(*args, **kwargs):
            await asyncio.sleep(5)
            return await original_process(*args, **kwargs)

        adapter._process_single_item = slow_process

        items = [
            {"id": "item1", "confidence": 0.9, "metadata": {"source_id": "item1"}},
            {"id": "item2", "confidence": 0.9, "metadata": {"source_id": "item2"}},
        ]
        tc = BatchTimeoutConfig(
            per_item_timeout_seconds=0.05,
            fail_fast_on_timeout=True,
        )

        result = await adapter.sync_validations_from_km(items, timeout_config=tc)

        # Only first item should be analyzed (batch stops after first timeout)
        assert result["records_analyzed"] == 1
        assert "fail_fast_on_timeout" in result["errors"][-1]

    @pytest.mark.asyncio
    async def test_total_batch_timeout(self):
        """Should stop batch when total timeout is exceeded."""
        from aragora.knowledge.mound.adapters._reverse_flow_base import (
            BatchTimeoutConfig,
        )

        records = {f"item{i}": {"id": f"item{i}"} for i in range(10)}
        adapter = self._make_adapter(records)

        original_process = adapter._process_single_item

        async def moderate_process(*args, **kwargs):
            await asyncio.sleep(0.02)  # Each item takes 20ms
            return await original_process(*args, **kwargs)

        adapter._process_single_item = moderate_process

        items = [
            {"id": f"item{i}", "confidence": 0.9, "metadata": {"source_id": f"item{i}"}}
            for i in range(10)
        ]
        tc = BatchTimeoutConfig(
            per_item_timeout_seconds=5.0,
            total_batch_timeout_seconds=0.05,  # 50ms total
        )

        result = await adapter.sync_validations_from_km(items, timeout_config=tc)

        # Should not have processed all items
        assert result["records_analyzed"] < 10
        assert any("Total batch timeout" in e for e in result["errors"])


# =============================================================================
# Deadlock Retry Tests
# =============================================================================


class TestDeadlockRetry:
    """Tests for deadlock-aware transaction retry."""

    def test_deadlock_detection(self):
        """Should correctly identify deadlock errors."""
        from aragora.knowledge.mound.resilience import TransactionManager

        pool = MagicMock()
        tm = TransactionManager(pool)

        assert tm._is_deadlock_error(Exception("deadlock detected"))
        assert tm._is_deadlock_error(Exception("ERROR 40P01: deadlock"))
        assert tm._is_deadlock_error(Exception("40001 serialization failure"))
        assert not tm._is_deadlock_error(Exception("connection refused"))
        assert not tm._is_deadlock_error(ValueError("invalid value"))

    def test_deadlock_delay_calculation(self):
        """Should calculate exponential backoff with jitter."""
        from aragora.knowledge.mound.resilience import (
            TransactionConfig,
            TransactionManager,
        )

        config = TransactionConfig(
            deadlock_base_delay=0.1,
            deadlock_max_delay=2.0,
        )
        pool = MagicMock()
        tm = TransactionManager(pool, config)

        # Attempt 0: ~0.1s (base)
        delay0 = tm._calculate_deadlock_delay(0)
        assert 0.05 <= delay0 <= 0.15  # 0.1 * (0.75 to 1.25)

        # Attempt 2: ~0.4s (0.1 * 2^2)
        delay2 = tm._calculate_deadlock_delay(2)
        assert 0.2 <= delay2 <= 0.6

        # Attempt 10: capped at max_delay
        delay10 = tm._calculate_deadlock_delay(10)
        assert delay10 <= 2.5  # max_delay * 1.25

    def test_transaction_config_has_deadlock_fields(self):
        """Should include deadlock config fields."""
        from aragora.knowledge.mound.resilience import TransactionConfig

        config = TransactionConfig()
        assert config.deadlock_retries == 3
        assert config.deadlock_base_delay == 0.1
        assert config.deadlock_max_delay == 2.0

    def test_deadlock_error_class(self):
        """Should create DeadlockError with retry count."""
        from aragora.knowledge.mound.resilience import DeadlockError

        err = DeadlockError("test deadlock", retry_count=3)
        assert err.retry_count == 3
        assert "test deadlock" in str(err)

    def test_stats_include_deadlock_retries(self):
        """Stats should include deadlock retry count."""
        from aragora.knowledge.mound.resilience import TransactionManager

        pool = MagicMock()
        tm = TransactionManager(pool)
        stats = tm.get_stats()
        assert "deadlock_retries" in stats


# =============================================================================
# Circuit Breaker Integration Tests
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with coordinator."""

    @pytest.mark.asyncio
    async def test_coordinator_skips_open_circuit(self):
        """Should skip adapter when circuit is open."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            AdapterRegistration,
            BidirectionalCoordinator,
            CoordinatorConfig,
        )

        config = CoordinatorConfig(
            check_circuit_breaker=True,
            skip_open_circuits=True,
        )
        coordinator = BidirectionalCoordinator(config=config)

        # Mock adapter
        adapter = MagicMock()
        adapter.sync_to_km = AsyncMock(return_value={"items_processed": 5})
        reg = AdapterRegistration(
            name="test_adapter",
            adapter=adapter,
            forward_method="sync_to_km",
            enabled=True,
        )

        # Mock circuit breaker to be open
        with patch(
            "aragora.knowledge.mound.bidirectional_coordinator.BidirectionalCoordinator._check_adapter_circuit"
        ) as mock_check:
            mock_check.return_value = (False, "Circuit open for test_adapter")

            result = await coordinator._sync_adapter_forward(reg)

        assert not result.success
        assert "Skipped" in result.errors[0]
        adapter.sync_to_km.assert_not_called()

    @pytest.mark.asyncio
    async def test_coordinator_allows_closed_circuit(self):
        """Should allow adapter when circuit is closed."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            AdapterRegistration,
            BidirectionalCoordinator,
            CoordinatorConfig,
        )

        config = CoordinatorConfig(
            check_circuit_breaker=True,
            skip_open_circuits=True,
        )
        coordinator = BidirectionalCoordinator(config=config)

        # Mock adapter
        adapter = MagicMock()
        adapter.sync_to_km = AsyncMock(return_value={"items_processed": 5})
        reg = AdapterRegistration(
            name="test_adapter",
            adapter=adapter,
            forward_method="sync_to_km",
            enabled=True,
        )

        with patch(
            "aragora.knowledge.mound.bidirectional_coordinator.BidirectionalCoordinator._check_adapter_circuit"
        ) as mock_check:
            mock_check.return_value = (True, "Circuit closed")

            result = await coordinator._sync_adapter_forward(reg)

        assert result.success
        adapter.sync_to_km.assert_called_once()

    def test_coordinator_config_has_circuit_fields(self):
        """Should include circuit breaker config fields."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            CoordinatorConfig,
        )

        config = CoordinatorConfig()
        assert config.check_circuit_breaker is True
        assert config.skip_open_circuits is True


# =============================================================================
# Per-Adapter Config Tests
# =============================================================================


class TestAdapterCircuitConfig:
    """Tests for per-adapter circuit breaker configuration."""

    def test_adapter_configs_exist(self):
        """Should have configs for known adapters."""
        from aragora.knowledge.mound.adapters._base import ADAPTER_CIRCUIT_CONFIGS

        expected_adapters = [
            "elo",
            "ranking",
            "evidence",
            "pulse",
            "continuum",
            "consensus",
            "critique",
            "insights",
            "belief",
            "cost",
            "control_plane",
            "receipt",
            "culture",
            "rlm",
        ]
        for name in expected_adapters:
            assert name in ADAPTER_CIRCUIT_CONFIGS, f"Missing config for {name}"

    def test_fast_adapters_have_tight_thresholds(self):
        """Fast adapters should have lower failure thresholds."""
        from aragora.knowledge.mound.adapters._base import ADAPTER_CIRCUIT_CONFIGS

        elo_config = ADAPTER_CIRCUIT_CONFIGS["elo"]
        assert elo_config.failure_threshold == 3
        assert elo_config.timeout_seconds == 15.0

    def test_slow_adapters_have_lenient_thresholds(self):
        """External/slow adapters should have higher thresholds and longer timeouts."""
        from aragora.knowledge.mound.adapters._base import ADAPTER_CIRCUIT_CONFIGS

        evidence_config = ADAPTER_CIRCUIT_CONFIGS["evidence"]
        assert evidence_config.failure_threshold == 5
        assert evidence_config.timeout_seconds == 60.0


# =============================================================================
# Health-Aware Circuit Breaker Tests
# =============================================================================


class TestHealthAwareCircuitBreaker:
    """Tests for HealthAwareCircuitBreaker."""

    def test_opens_circuit_on_unhealthy(self):
        """Should open circuit when health monitor reports unhealthy."""
        from aragora.knowledge.mound.resilience import (
            HealthAwareCircuitBreaker,
        )

        health_monitor = MagicMock()
        health_monitor.is_healthy.return_value = False

        hcb = HealthAwareCircuitBreaker("test", health_monitor)

        # Initially closed, but health check should open it
        assert not hcb.can_proceed()

    def test_allows_when_healthy(self):
        """Should allow requests when health monitor reports healthy."""
        from aragora.knowledge.mound.resilience import (
            HealthAwareCircuitBreaker,
        )

        health_monitor = MagicMock()
        health_monitor.is_healthy.return_value = True

        hcb = HealthAwareCircuitBreaker("test", health_monitor)
        assert hcb.can_proceed()

    def test_stats_include_health_status(self):
        """Stats should include health monitor status."""
        from aragora.knowledge.mound.resilience import (
            HealthAwareCircuitBreaker,
        )

        health_monitor = MagicMock()
        health_monitor.is_healthy.return_value = True

        hcb = HealthAwareCircuitBreaker("test", health_monitor)
        stats = hcb.get_stats()

        assert "health_monitor_healthy" in stats
        assert stats["health_monitor_healthy"] is True


# =============================================================================
# Latency Tracker Tests
# =============================================================================


class TestLatencyTracker:
    """Tests for LatencyTracker and adaptive timeouts."""

    def test_empty_tracker_returns_min_timeout(self):
        """Should return min timeout when no samples recorded."""
        from aragora.knowledge.mound.resilience import LatencyTracker

        tracker = LatencyTracker(min_timeout_ms=100.0)
        assert tracker.get_adaptive_timeout() == 100.0

    def test_adaptive_timeout_based_on_p95(self):
        """Should compute adaptive timeout from P95 * multiplier."""
        from aragora.knowledge.mound.resilience import LatencyTracker

        tracker = LatencyTracker(timeout_multiplier=1.5)

        # Record 100 samples: 1ms to 100ms
        for i in range(1, 101):
            tracker.record(float(i))

        timeout = tracker.get_adaptive_timeout()
        # P95 should be around 95ms, * 1.5 = ~142.5ms
        assert 100.0 <= timeout <= 200.0

    def test_timeout_clamped_to_max(self):
        """Should not exceed max timeout."""
        from aragora.knowledge.mound.resilience import LatencyTracker

        tracker = LatencyTracker(
            timeout_multiplier=10.0,
            max_timeout_ms=500.0,
        )

        for _ in range(100):
            tracker.record(1000.0)  # 1000ms samples

        timeout = tracker.get_adaptive_timeout()
        assert timeout == 500.0  # Clamped to max

    def test_stats_output(self):
        """Should provide comprehensive stats."""
        from aragora.knowledge.mound.resilience import LatencyTracker

        tracker = LatencyTracker()
        for i in range(50):
            tracker.record(float(i * 10))

        stats = tracker.get_stats()
        assert "p50_ms" in stats
        assert "p90_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "adaptive_timeout_ms" in stats
        assert stats["sample_count"] == 50

    def test_sample_window_trimming(self):
        """Should trim old samples when exceeding max."""
        from aragora.knowledge.mound.resilience import LatencyTracker

        tracker = LatencyTracker(max_samples=10)

        for i in range(100):
            tracker.record(float(i))

        stats = tracker.get_stats()
        assert stats["sample_count"] == 10
        assert stats["total_count"] == 100

    def test_percentile_calculation(self):
        """Should calculate percentiles correctly."""
        from aragora.knowledge.mound.resilience import LatencyTracker

        tracker = LatencyTracker()
        # Record sorted values 1-100
        for i in range(1, 101):
            tracker.record(float(i))

        p50 = tracker.get_percentile(0.50)
        p99 = tracker.get_percentile(0.99)

        assert 45 <= p50 <= 55
        assert 95 <= p99 <= 100
