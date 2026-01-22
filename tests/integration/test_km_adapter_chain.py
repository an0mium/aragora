"""
Integration tests for Knowledge Mound adapter chain sync.

Tests the complete adapter chain including:
1. Factory creating multiple adapters
2. Coordinator orchestrating sync across adapters
3. Bidirectional sync consistency
4. Multi-adapter data flow validation
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MockMemoryItem:
    """Mock memory item for testing."""

    id: str
    content: str
    confidence: float = 0.8
    tier: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockEloMatch:
    """Mock ELO match result."""

    agent_a: str
    agent_b: str
    winner: str
    elo_delta: float = 16.0


class MockContinuumMemory:
    """Mock Continuum memory for testing adapter chain."""

    def __init__(self):
        self.items: Dict[str, MockMemoryItem] = {}
        self.sync_calls = 0

    async def store(self, item_id: str, content: str, **kwargs) -> str:
        self.items[item_id] = MockMemoryItem(id=item_id, content=content, **kwargs)
        return item_id

    async def retrieve(self, item_id: str) -> Optional[MockMemoryItem]:
        return self.items.get(item_id)

    async def list_items(self, limit: int = 100) -> List[MockMemoryItem]:
        return list(self.items.values())[:limit]


class MockEloSystem:
    """Mock ELO system for testing adapter chain."""

    def __init__(self):
        self.ratings: Dict[str, float] = {"agent-1": 1000, "agent-2": 1000}
        self.matches: List[MockEloMatch] = []

    def get_rating(self, agent_id: str) -> float:
        return self.ratings.get(agent_id, 1000)

    def update_ratings(self, agent_a: str, agent_b: str, winner: str) -> None:
        delta = 16.0
        if winner == agent_a:
            self.ratings[agent_a] = self.ratings.get(agent_a, 1000) + delta
            self.ratings[agent_b] = self.ratings.get(agent_b, 1000) - delta
        else:
            self.ratings[agent_a] = self.ratings.get(agent_a, 1000) - delta
            self.ratings[agent_b] = self.ratings.get(agent_b, 1000) + delta

        self.matches.append(MockEloMatch(agent_a, agent_b, winner))


class MockCritiqueStore:
    """Mock critique store for testing adapter chain."""

    def __init__(self):
        self.critiques: Dict[str, Dict[str, Any]] = {}

    async def store_critique(self, critique_id: str, **kwargs) -> str:
        self.critiques[critique_id] = kwargs
        return critique_id

    async def get_critiques(self, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self.critiques.values())[:limit]


class TestAdapterFactoryChain:
    """Test factory creating and connecting multiple adapters."""

    def test_factory_creates_multiple_adapters(self):
        """Factory should create all adapters with available deps."""
        from aragora.knowledge.mound.adapters import AdapterFactory

        mock_continuum = MockContinuumMemory()
        mock_elo = MockEloSystem()
        mock_critique = MockCritiqueStore()

        factory = AdapterFactory()
        adapters = factory.create_from_subsystems(
            continuum_memory=mock_continuum,
            elo_system=mock_elo,
            memory=mock_critique,  # CritiqueStore
        )

        # Should have created adapters for each dep
        assert "continuum" in adapters
        assert "elo" in adapters
        assert "critique" in adapters
        assert "belief" in adapters  # No deps required

    def test_factory_with_coordinator_registration(self):
        """Factory should register adapters with coordinator."""
        from aragora.knowledge.mound.adapters import AdapterFactory
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        mock_elo = MockEloSystem()

        factory = AdapterFactory()
        adapters = factory.create_from_subsystems(elo_system=mock_elo)

        coordinator = BidirectionalCoordinator()
        registered = factory.register_with_coordinator(coordinator, adapters)

        assert registered >= 2  # At least elo and belief
        status = coordinator.get_status()
        assert status["total_adapters"] >= 2


class TestCoordinatorMultiAdapterSync:
    """Test coordinator orchestrating sync across multiple adapters."""

    @pytest.fixture
    def mock_adapters(self):
        """Create mock adapters for testing."""

        class TrackedAdapter:
            def __init__(self, name: str):
                self.name = name
                self.forward_count = 0
                self.reverse_count = 0
                self.forward_data = []
                self.reverse_data = []

            async def sync_to_km(self):
                self.forward_count += 1
                return {"items_synced": 5, "adapter": self.name}

            async def sync_from_km(self, items, min_confidence=0.7):
                self.reverse_count += 1
                self.reverse_data = items
                return {"items_updated": len(items) if items else 0}

        return {
            "continuum": TrackedAdapter("continuum"),
            "elo": TrackedAdapter("elo"),
            "critique": TrackedAdapter("critique"),
        }

    @pytest.mark.asyncio
    async def test_forward_sync_all_adapters(self, mock_adapters):
        """Coordinator should sync all registered adapters."""
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        coordinator = BidirectionalCoordinator()

        for name, adapter in mock_adapters.items():
            coordinator.register_adapter(
                name=name,
                adapter=adapter,
                forward_method="sync_to_km",
                reverse_method="sync_from_km",
                priority=50,
            )

        # Run forward sync
        results = await coordinator.sync_all_to_km()

        # All adapters should have been called
        for name, adapter in mock_adapters.items():
            assert adapter.forward_count == 1

    @pytest.mark.asyncio
    async def test_sync_respects_priority_order(self, mock_adapters):
        """Higher priority adapters should sync first."""
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        coordinator = BidirectionalCoordinator()

        # Register with different priorities
        priorities = {"continuum": 100, "elo": 50, "critique": 25}
        sync_order = []

        for name, adapter in mock_adapters.items():
            # Track order via side effect
            original_sync = adapter.sync_to_km

            async def tracked_sync(n=name):
                sync_order.append(n)
                return {"items_synced": 1}

            adapter.sync_to_km = tracked_sync

            coordinator.register_adapter(
                name=name,
                adapter=adapter,
                forward_method="sync_to_km",
                priority=priorities[name],
            )

        await coordinator.sync_all_to_km()

        # Verify order: highest priority first
        assert sync_order[0] == "continuum"  # 100
        assert sync_order[1] == "elo"  # 50
        assert sync_order[2] == "critique"  # 25

    @pytest.mark.asyncio
    async def test_adapter_failure_isolation(self, mock_adapters):
        """One adapter failure should not stop others."""
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        coordinator = BidirectionalCoordinator()

        # Make one adapter fail
        async def failing_sync():
            raise ValueError("Simulated failure")

        mock_adapters["elo"].sync_to_km = failing_sync

        for name, adapter in mock_adapters.items():
            coordinator.register_adapter(
                name=name,
                adapter=adapter,
                forward_method="sync_to_km",
            )

        # Should complete without raising
        results = await coordinator.sync_all_to_km()

        # Other adapters should still have synced
        assert mock_adapters["continuum"].forward_count == 1
        assert mock_adapters["critique"].forward_count == 1


class TestBidirectionalSyncConsistency:
    """Test bidirectional sync maintains data consistency."""

    @pytest.mark.asyncio
    async def test_round_trip_consistency(self):
        """Data synced forward should be retrievable in reverse."""
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        # Track data flow
        forward_data = []
        reverse_received = []

        class ConsistentAdapter:
            async def sync_to_km(self):
                data = [{"id": "item-1", "content": "test"}]
                forward_data.extend(data)
                return {"items": data}

            async def sync_from_km(self, items, min_confidence=0.7):
                reverse_received.extend(items or [])
                return {"processed": len(items or [])}

        adapter = ConsistentAdapter()
        coordinator = BidirectionalCoordinator()

        coordinator.register_adapter(
            name="test",
            adapter=adapter,
            forward_method="sync_to_km",
            reverse_method="sync_from_km",
        )

        # Forward sync
        await coordinator.sync_all_to_km()

        # Reverse sync with the forward data
        await coordinator.sync_all_from_km(forward_data)

        # Data should have flowed through
        assert len(forward_data) == 1
        assert len(reverse_received) == 1

    @pytest.mark.asyncio
    async def test_confidence_filtering_on_reverse(self):
        """Reverse sync should respect confidence threshold."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
            CoordinatorConfig,
        )

        received_confidence = []

        class ConfidenceAdapter:
            async def sync_to_km(self):
                return {}

            async def sync_from_km(self, items, min_confidence=0.7):
                received_confidence.append(min_confidence)
                return {}

        config = CoordinatorConfig(min_confidence_for_reverse=0.85)
        coordinator = BidirectionalCoordinator(config=config)

        coordinator.register_adapter(
            name="test",
            adapter=ConfidenceAdapter(),
            forward_method="sync_to_km",
            reverse_method="sync_from_km",
        )

        # Pass non-empty items to trigger reverse sync
        test_items = [{"id": "item-1", "confidence": 0.9}]
        await coordinator.sync_all_from_km(test_items)

        # Should have used configured confidence threshold
        assert received_confidence[0] == 0.85


class TestAdapterChainDataFlow:
    """Test data flow through complete adapter chain."""

    def test_factory_to_coordinator_data_flow(self):
        """Test complete flow: factory → adapters → coordinator."""
        from aragora.knowledge.mound.adapters import AdapterFactory
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        # Create mock subsystems
        mock_elo = MagicMock()
        mock_elo.get_rating = MagicMock(return_value=1500)

        events_received = []

        def track_event(event_type, data):
            events_received.append((event_type, data))

        # Create factory with event tracking
        factory = AdapterFactory(event_callback=track_event)
        adapters = factory.create_from_subsystems(elo_system=mock_elo)

        # Register with coordinator
        coordinator = BidirectionalCoordinator()
        registered = factory.register_with_coordinator(coordinator, adapters)

        # Verify chain is connected
        assert registered >= 2
        status = coordinator.get_status()
        assert "elo" in status["adapters"]

    @pytest.mark.asyncio
    async def test_multi_system_sync_report(self):
        """Test sync report aggregates results from all adapters."""
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        # Create adapters with different results
        adapters = {}
        for name, count in [("continuum", 10), ("elo", 5), ("critique", 3)]:

            class CountingAdapter:
                def __init__(self, n):
                    self.count = n

                async def sync_to_km(self):
                    return {"items_synced": self.count}

            adapters[name] = CountingAdapter(count)

        coordinator = BidirectionalCoordinator()
        for name, adapter in adapters.items():
            coordinator.register_adapter(
                name=name,
                adapter=adapter,
                forward_method="sync_to_km",
            )

        results = await coordinator.sync_all_to_km()

        # Should have results from all adapters (list of SyncResult)
        assert len(results) == 3

        # All should be successful
        for result in results:
            assert result.success


class TestAdapterEnableDisable:
    """Test enabling/disabling adapters in the chain."""

    def test_disabled_adapter_not_synced(self):
        """Disabled adapters should not be included in sync."""
        from aragora.knowledge.mound.adapters import AdapterFactory
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        mock_cost = MagicMock()

        factory = AdapterFactory()
        adapters = factory.create_from_subsystems(cost_tracker=mock_cost)

        coordinator = BidirectionalCoordinator()
        factory.register_with_coordinator(coordinator, adapters)

        # Cost adapter should be disabled by default
        status = coordinator.get_status()
        if "cost" in status["adapters"]:
            assert status["adapters"]["cost"]["enabled"] is False

    def test_enable_adapter_at_runtime(self):
        """Adapters can be enabled at runtime."""
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        class TestAdapter:
            async def sync_to_km(self):
                return {}

        coordinator = BidirectionalCoordinator()
        coordinator.register_adapter(
            name="test",
            adapter=TestAdapter(),
            forward_method="sync_to_km",
        )

        # Disable then re-enable
        coordinator.disable_adapter("test")
        status = coordinator.get_status()
        assert status["adapters"]["test"]["enabled"] is False

        coordinator.enable_adapter("test")
        status = coordinator.get_status()
        assert status["adapters"]["test"]["enabled"] is True


class TestConcurrentAdapterSync:
    """Test concurrent sync behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_forward_sync(self):
        """Multiple adapters should sync concurrently when enabled."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
            CoordinatorConfig,
        )
        import time

        start_times = {}
        end_times = {}

        class SlowAdapter:
            def __init__(self, name, delay):
                self.name = name
                self.delay = delay

            async def sync_to_km(self):
                import asyncio

                start_times[self.name] = time.time()
                await asyncio.sleep(self.delay)
                end_times[self.name] = time.time()
                return {}

        # Enable parallel sync
        config = CoordinatorConfig(parallel_sync=True)
        coordinator = BidirectionalCoordinator(config=config)

        # Register adapters with same priority (should run in parallel)
        for name in ["a", "b", "c"]:
            coordinator.register_adapter(
                name=name,
                adapter=SlowAdapter(name, 0.1),
                forward_method="sync_to_km",
                priority=50,  # Same priority
            )

        await coordinator.sync_all_to_km()

        # All should have started around the same time (within 50ms)
        start_spread = max(start_times.values()) - min(start_times.values())
        assert start_spread < 0.05  # Concurrent start


class TestAdapterMetrics:
    """Test adapter metrics collection."""

    @pytest.mark.asyncio
    async def test_sync_metrics_recorded(self):
        """Sync operations should record metrics."""
        from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

        class MetricAdapter:
            async def sync_to_km(self):
                return {"items_synced": 10}

        coordinator = BidirectionalCoordinator()
        coordinator.register_adapter(
            name="metric_test",
            adapter=MetricAdapter(),
            forward_method="sync_to_km",
        )

        await coordinator.sync_all_to_km()

        # Get status (includes metrics)
        status = coordinator.get_status()
        assert status["total_forward_syncs"] >= 1
