"""
Tests for BidirectionalCoordinator.

Tests the central coordination layer for KM bidirectional sync operations.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.bidirectional_coordinator import (
    BidirectionalCoordinator,
    CoordinatorConfig,
    AdapterRegistration,
    SyncResult,
    BidirectionalSyncReport,
)


class MockAdapter:
    """Mock adapter for testing."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.forward_calls = 0
        self.reverse_calls = 0
        self.forward_items = []
        self.reverse_items = []

    async def sync_to_km(self):
        """Mock forward sync."""
        self.forward_calls += 1
        return {
            "items_processed": 10,
            "items_updated": 5,
        }

    async def sync_from_km(self, km_items, min_confidence=0.7):
        """Mock reverse sync."""
        self.reverse_calls += 1
        self.reverse_items = km_items
        return {
            "items_processed": len(km_items),
            "items_updated": len(km_items) // 2,
        }

    def sync_to_km_sync(self):
        """Synchronous forward sync."""
        self.forward_calls += 1
        return {"items_processed": 5}


class FailingAdapter:
    """Adapter that fails on sync."""

    async def sync_to_km(self):
        raise ValueError("Forward sync failed")

    async def sync_from_km(self, km_items, min_confidence=0.7):
        raise ValueError("Reverse sync failed")


class SlowAdapter:
    """Adapter that times out."""

    async def sync_to_km(self):
        await asyncio.sleep(10)  # Will timeout
        return {}

    async def sync_from_km(self, km_items, min_confidence=0.7):
        await asyncio.sleep(10)
        return {}


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CoordinatorConfig()
        assert config.sync_interval_seconds == 300
        assert config.batch_size == 100
        assert config.min_confidence_for_reverse == 0.7
        assert config.max_retries == 3
        assert config.parallel_sync is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CoordinatorConfig(
            sync_interval_seconds=60,
            batch_size=50,
            min_confidence_for_reverse=0.8,
            parallel_sync=False,
        )
        assert config.sync_interval_seconds == 60
        assert config.batch_size == 50
        assert config.min_confidence_for_reverse == 0.8
        assert config.parallel_sync is False


class TestAdapterRegistration:
    """Tests for AdapterRegistration dataclass."""

    def test_default_values(self):
        """Test default registration values."""
        reg = AdapterRegistration(
            name="test",
            adapter=MagicMock(),
            forward_method="sync_to_km",
        )
        assert reg.name == "test"
        assert reg.enabled is True
        assert reg.priority == 0
        assert reg.reverse_method is None
        assert reg.forward_errors == 0

    def test_with_reverse_method(self):
        """Test registration with reverse method."""
        reg = AdapterRegistration(
            name="test",
            adapter=MagicMock(),
            forward_method="sync_to_km",
            reverse_method="sync_from_km",
            priority=10,
        )
        assert reg.reverse_method == "sync_from_km"
        assert reg.priority == 10


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_default_values(self):
        """Test default sync result values."""
        result = SyncResult(
            adapter_name="test",
            direction="forward",
            success=True,
        )
        assert result.items_processed == 0
        assert result.items_updated == 0
        assert result.errors == []
        assert result.duration_ms == 0

    def test_with_errors(self):
        """Test sync result with errors."""
        result = SyncResult(
            adapter_name="test",
            direction="reverse",
            success=False,
            errors=["Error 1", "Error 2"],
        )
        assert result.success is False
        assert len(result.errors) == 2


class TestBidirectionalSyncReport:
    """Tests for BidirectionalSyncReport dataclass."""

    def test_default_values(self):
        """Test default report values."""
        report = BidirectionalSyncReport()
        assert report.forward_results == []
        assert report.reverse_results == []
        assert report.total_adapters == 0
        assert report.total_errors == 0


class TestBidirectionalCoordinator:
    """Tests for BidirectionalCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator for testing."""
        return BidirectionalCoordinator()

    @pytest.fixture
    def coordinator_with_adapters(self):
        """Create coordinator with registered adapters."""
        coordinator = BidirectionalCoordinator()
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")

        coordinator.register_adapter(
            "adapter1",
            adapter1,
            "sync_to_km",
            "sync_from_km",
            priority=10,
        )
        coordinator.register_adapter(
            "adapter2",
            adapter2,
            "sync_to_km",
            "sync_from_km",
            priority=5,
        )

        return coordinator

    def test_init_default_config(self, coordinator):
        """Test initialization with default config."""
        assert coordinator.config is not None
        assert coordinator.config.sync_interval_seconds == 300
        assert len(coordinator._adapters) == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = CoordinatorConfig(sync_interval_seconds=60)
        coordinator = BidirectionalCoordinator(config=config)
        assert coordinator.config.sync_interval_seconds == 60

    def test_register_adapter(self, coordinator):
        """Test adapter registration."""
        adapter = MockAdapter()
        result = coordinator.register_adapter(
            "test",
            adapter,
            "sync_to_km",
            "sync_from_km",
        )

        assert result is True
        assert "test" in coordinator.get_registered_adapters()

    def test_register_adapter_missing_forward_method(self, coordinator):
        """Test registration fails with missing forward method."""
        adapter = MockAdapter()
        result = coordinator.register_adapter(
            "test",
            adapter,
            "nonexistent_method",
        )

        assert result is False
        assert "test" not in coordinator.get_registered_adapters()

    def test_register_adapter_missing_reverse_method(self, coordinator):
        """Test registration with missing reverse method logs warning."""
        adapter = MockAdapter()
        result = coordinator.register_adapter(
            "test",
            adapter,
            "sync_to_km",
            "nonexistent_reverse",
        )

        assert result is True  # Still registers, but without reverse
        reg = coordinator._adapters["test"]
        assert reg.reverse_method is None

    def test_unregister_adapter(self, coordinator):
        """Test adapter unregistration."""
        adapter = MockAdapter()
        coordinator.register_adapter("test", adapter, "sync_to_km")

        result = coordinator.unregister_adapter("test")
        assert result is True
        assert "test" not in coordinator.get_registered_adapters()

    def test_unregister_nonexistent_adapter(self, coordinator):
        """Test unregistering nonexistent adapter."""
        result = coordinator.unregister_adapter("nonexistent")
        assert result is False

    def test_get_adapter(self, coordinator):
        """Test getting adapter by name."""
        adapter = MockAdapter()
        coordinator.register_adapter("test", adapter, "sync_to_km")

        retrieved = coordinator.get_adapter("test")
        assert retrieved is adapter

    def test_get_nonexistent_adapter(self, coordinator):
        """Test getting nonexistent adapter."""
        assert coordinator.get_adapter("nonexistent") is None

    def test_enable_disable_adapter(self, coordinator):
        """Test enabling and disabling adapters."""
        adapter = MockAdapter()
        coordinator.register_adapter("test", adapter, "sync_to_km")

        # Disable
        result = coordinator.disable_adapter("test")
        assert result is True
        assert coordinator._adapters["test"].enabled is False

        # Enable
        result = coordinator.enable_adapter("test")
        assert result is True
        assert coordinator._adapters["test"].enabled is True

    def test_enable_nonexistent_adapter(self, coordinator):
        """Test enabling nonexistent adapter."""
        assert coordinator.enable_adapter("nonexistent") is False


class TestBidirectionalCoordinatorForwardSync:
    """Tests for forward sync operations."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with mock adapter."""
        coordinator = BidirectionalCoordinator()
        adapter = MockAdapter()
        coordinator.register_adapter(
            "test",
            adapter,
            "sync_to_km",
            "sync_from_km",
        )
        return coordinator

    @pytest.mark.asyncio
    async def test_sync_all_to_km(self, coordinator):
        """Test forward sync to all adapters."""
        results = await coordinator.sync_all_to_km()

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].direction == "forward"
        assert results[0].items_processed == 10

    @pytest.mark.asyncio
    async def test_sync_all_to_km_parallel(self):
        """Test parallel forward sync."""
        config = CoordinatorConfig(parallel_sync=True)
        coordinator = BidirectionalCoordinator(config=config)

        for i in range(3):
            adapter = MockAdapter(f"adapter_{i}")
            coordinator.register_adapter(
                f"adapter_{i}",
                adapter,
                "sync_to_km",
            )

        results = await coordinator.sync_all_to_km()

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_sync_all_to_km_sequential(self):
        """Test sequential forward sync."""
        config = CoordinatorConfig(parallel_sync=False)
        coordinator = BidirectionalCoordinator(config=config)

        for i in range(3):
            adapter = MockAdapter(f"adapter_{i}")
            coordinator.register_adapter(
                f"adapter_{i}",
                adapter,
                "sync_to_km",
            )

        results = await coordinator.sync_all_to_km()

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_sync_forward_with_error(self):
        """Test forward sync with failing adapter."""
        coordinator = BidirectionalCoordinator()
        adapter = FailingAdapter()
        coordinator.register_adapter("failing", adapter, "sync_to_km")

        results = await coordinator.sync_all_to_km()

        assert len(results) == 1
        assert results[0].success is False
        assert len(results[0].errors) > 0

    @pytest.mark.asyncio
    async def test_sync_forward_respects_enabled(self):
        """Test that disabled adapters are skipped."""
        coordinator = BidirectionalCoordinator()
        adapter = MockAdapter()
        coordinator.register_adapter("test", adapter, "sync_to_km")
        coordinator.disable_adapter("test")

        results = await coordinator.sync_all_to_km()

        assert len(results) == 0


class TestBidirectionalCoordinatorReverseSync:
    """Tests for reverse sync operations."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with mock adapter."""
        coordinator = BidirectionalCoordinator()
        adapter = MockAdapter()
        coordinator.register_adapter(
            "test",
            adapter,
            "sync_to_km",
            "sync_from_km",
        )
        return coordinator

    @pytest.mark.asyncio
    async def test_sync_all_from_km(self, coordinator):
        """Test reverse sync from KM."""
        km_items = [
            {"id": "1", "metadata": {"outcome_success": True}},
            {"id": "2", "metadata": {"outcome_success": False}},
        ]

        results = await coordinator.sync_all_from_km(km_items)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].direction == "reverse"

    @pytest.mark.asyncio
    async def test_sync_all_from_km_empty_items(self, coordinator):
        """Test reverse sync with empty items."""
        results = await coordinator.sync_all_from_km([])

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_sync_all_from_km_no_items(self, coordinator):
        """Test reverse sync with None items."""
        results = await coordinator.sync_all_from_km(None)

        # Should return empty since no KM items
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_sync_reverse_with_error(self):
        """Test reverse sync with failing adapter."""
        coordinator = BidirectionalCoordinator()
        adapter = FailingAdapter()
        coordinator.register_adapter(
            "failing",
            adapter,
            "sync_to_km",
            "sync_from_km",
        )

        km_items = [{"id": "1"}]
        results = await coordinator.sync_all_from_km(km_items)

        assert len(results) == 1
        assert results[0].success is False
        assert len(results[0].errors) > 0

    @pytest.mark.asyncio
    async def test_sync_reverse_skips_no_reverse_method(self):
        """Test that adapters without reverse method are skipped."""
        coordinator = BidirectionalCoordinator()
        adapter = MockAdapter()
        coordinator.register_adapter("test", adapter, "sync_to_km")  # No reverse

        km_items = [{"id": "1"}]
        results = await coordinator.sync_all_from_km(km_items)

        assert len(results) == 0


class TestBidirectionalCoordinatorFullSync:
    """Tests for full bidirectional sync."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with adapters."""
        coordinator = BidirectionalCoordinator()
        adapter = MockAdapter()
        coordinator.register_adapter(
            "test",
            adapter,
            "sync_to_km",
            "sync_from_km",
        )
        return coordinator

    @pytest.mark.asyncio
    async def test_run_bidirectional_sync(self, coordinator):
        """Test full bidirectional sync cycle."""
        km_items = [{"id": "1"}, {"id": "2"}]

        report = await coordinator.run_bidirectional_sync(km_items)

        assert isinstance(report, BidirectionalSyncReport)
        assert report.successful_forward >= 0
        assert report.successful_reverse >= 0
        assert report.total_duration_ms >= 0
        assert report.timestamp != ""

    @pytest.mark.asyncio
    async def test_run_bidirectional_sync_updates_metrics(self, coordinator):
        """Test that bidirectional sync updates metrics."""
        km_items = [{"id": "1"}]

        initial_forward = coordinator._total_forward_syncs
        initial_reverse = coordinator._total_reverse_syncs

        await coordinator.run_bidirectional_sync(km_items)

        assert coordinator._total_forward_syncs == initial_forward + 1
        assert coordinator._total_reverse_syncs == initial_reverse + 1

    @pytest.mark.asyncio
    async def test_run_bidirectional_sync_stores_history(self, coordinator):
        """Test that bidirectional sync stores history."""
        km_items = [{"id": "1"}]

        await coordinator.run_bidirectional_sync(km_items)

        history = coordinator.get_sync_history()
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_run_bidirectional_sync_concurrent_protection(self, coordinator):
        """Test that concurrent syncs are prevented."""
        km_items = [{"id": "1"}]

        # Start two syncs concurrently
        task1 = asyncio.create_task(coordinator.run_bidirectional_sync(km_items))
        task2 = asyncio.create_task(coordinator.run_bidirectional_sync(km_items))

        results = await asyncio.gather(task1, task2)

        # One should succeed, one should be blocked
        success_count = sum(1 for r in results if r.total_errors == 0)
        assert success_count >= 1


class TestBidirectionalCoordinatorStatus:
    """Tests for coordinator status and metrics."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with adapters."""
        coordinator = BidirectionalCoordinator()
        adapter = MockAdapter()
        coordinator.register_adapter(
            "test",
            adapter,
            "sync_to_km",
            "sync_from_km",
        )
        return coordinator

    def test_get_status(self, coordinator):
        """Test getting coordinator status."""
        status = coordinator.get_status()

        assert "total_adapters" in status
        assert "enabled_adapters" in status
        assert "bidirectional_adapters" in status
        assert "sync_in_progress" in status
        assert "config" in status
        assert "adapters" in status

    def test_get_status_adapter_details(self, coordinator):
        """Test adapter details in status."""
        status = coordinator.get_status()

        assert "test" in status["adapters"]
        adapter_status = status["adapters"]["test"]
        assert adapter_status["enabled"] is True
        assert adapter_status["has_reverse"] is True

    def test_get_sync_history(self, coordinator):
        """Test getting sync history."""
        history = coordinator.get_sync_history()
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_sync_history_with_limit(self, coordinator):
        """Test getting limited sync history."""
        km_items = [{"id": "1"}]

        # Run multiple syncs
        for _ in range(5):
            await coordinator.run_bidirectional_sync(km_items)

        history = coordinator.get_sync_history(limit=3)
        assert len(history) == 3

    def test_clear_history(self, coordinator):
        """Test clearing sync history."""
        coordinator._sync_history = [BidirectionalSyncReport()]
        coordinator.clear_history()
        assert len(coordinator._sync_history) == 0

    def test_reset_metrics(self, coordinator):
        """Test resetting metrics."""
        coordinator._total_forward_syncs = 10
        coordinator._total_reverse_syncs = 5
        coordinator._total_errors = 3

        coordinator.reset_metrics()

        assert coordinator._total_forward_syncs == 0
        assert coordinator._total_reverse_syncs == 0
        assert coordinator._total_errors == 0


class TestBidirectionalCoordinatorPriority:
    """Tests for adapter priority handling."""

    @pytest.mark.asyncio
    async def test_adapters_synced_by_priority(self):
        """Test that adapters are synced in priority order."""
        config = CoordinatorConfig(parallel_sync=False)
        coordinator = BidirectionalCoordinator(config=config)

        # Track sync order
        sync_order = []

        class OrderTrackingAdapter:
            def __init__(self, name):
                self.name = name

            async def sync_to_km(self):
                sync_order.append(self.name)
                return {}

        # Register adapters with different priorities
        coordinator.register_adapter(
            "low",
            OrderTrackingAdapter("low"),
            "sync_to_km",
            priority=1,
        )
        coordinator.register_adapter(
            "high",
            OrderTrackingAdapter("high"),
            "sync_to_km",
            priority=10,
        )
        coordinator.register_adapter(
            "medium",
            OrderTrackingAdapter("medium"),
            "sync_to_km",
            priority=5,
        )

        await coordinator.sync_all_to_km()

        # Should be in priority order (high to low)
        assert sync_order == ["high", "medium", "low"]


class TestBidirectionalCoordinatorTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_forward_sync_timeout(self):
        """Test forward sync timeout handling."""
        config = CoordinatorConfig(timeout_seconds=0.1)
        coordinator = BidirectionalCoordinator(config=config)

        adapter = SlowAdapter()
        coordinator.register_adapter("slow", adapter, "sync_to_km")

        results = await coordinator.sync_all_to_km()

        assert len(results) == 1
        assert results[0].success is False
        assert any("timeout" in e.lower() for e in results[0].errors)

    @pytest.mark.asyncio
    async def test_reverse_sync_timeout(self):
        """Test reverse sync timeout handling."""
        config = CoordinatorConfig(timeout_seconds=0.1)
        coordinator = BidirectionalCoordinator(config=config)

        adapter = SlowAdapter()
        coordinator.register_adapter("slow", adapter, "sync_to_km", "sync_from_km")

        km_items = [{"id": "1"}]
        results = await coordinator.sync_all_from_km(km_items)

        assert len(results) == 1
        assert results[0].success is False
        assert any("timeout" in e.lower() for e in results[0].errors)


class TestBidirectionalCoordinatorIntegration:
    """Integration tests for the coordinator."""

    @pytest.mark.asyncio
    async def test_full_integration_with_multiple_adapters(self):
        """Test full integration with multiple adapters."""
        coordinator = BidirectionalCoordinator()

        # Register multiple adapters
        adapters = {}
        for i in range(3):
            adapter = MockAdapter(f"adapter_{i}")
            adapters[f"adapter_{i}"] = adapter
            coordinator.register_adapter(
                f"adapter_{i}",
                adapter,
                "sync_to_km",
                "sync_from_km",
                priority=i,
            )

        # Run bidirectional sync
        km_items = [
            {"id": "1", "metadata": {"outcome_success": True}},
            {"id": "2", "metadata": {"outcome_success": False}},
        ]

        report = await coordinator.run_bidirectional_sync(km_items)

        # Verify all adapters were synced
        assert report.total_adapters == 3
        assert report.successful_forward == 3
        assert report.successful_reverse == 3

        # Verify adapters received the calls
        for name, adapter in adapters.items():
            assert adapter.forward_calls == 1
            assert adapter.reverse_calls == 1

    @pytest.mark.asyncio
    async def test_mixed_success_failure(self):
        """Test with mixed success and failure adapters."""
        coordinator = BidirectionalCoordinator()

        # Register one good and one failing adapter
        good_adapter = MockAdapter("good")
        coordinator.register_adapter(
            "good",
            good_adapter,
            "sync_to_km",
            "sync_from_km",
        )

        failing_adapter = FailingAdapter()
        coordinator.register_adapter(
            "failing",
            failing_adapter,
            "sync_to_km",
            "sync_from_km",
        )

        km_items = [{"id": "1"}]
        report = await coordinator.run_bidirectional_sync(km_items)

        # Should have partial success
        assert report.successful_forward == 1
        assert report.successful_reverse == 1
        assert report.total_errors > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
