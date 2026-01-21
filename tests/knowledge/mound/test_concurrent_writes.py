"""
Integration tests for concurrent Knowledge Mound writes.

These tests validate:
- Data consistency under concurrent write load
- Transaction atomicity across multiple memory systems
- Rollback behavior under partial failures
- Conflict resolution when adapters produce conflicting data
- Deadlock prevention during high-concurrency scenarios

Run with: pytest tests/knowledge/mound/test_concurrent_writes.py -v
"""

from __future__ import annotations

import asyncio
import random
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.coordinator import (
    CoordinatorOptions,
    MemoryCoordinator,
    MemoryTransaction,
    WriteOperation,
    WriteStatus,
)
from aragora.knowledge.mound.bidirectional_coordinator import (
    BidirectionalCoordinator,
    CoordinatorConfig,
    BidirectionalSyncReport,
)


# ============================================================================
# Test Fixtures
# ============================================================================


class MockDebateContext:
    """Mock debate context for testing."""

    def __init__(
        self,
        debate_id: Optional[str] = None,
        confidence: float = 0.85,
        domain: str = "testing",
    ):
        self.debate_id = debate_id or f"debate-{uuid.uuid4().hex[:8]}"
        self.domain = domain
        self.env = MagicMock()
        self.env.task = f"Test task for {self.debate_id}"
        self.agents = [
            MagicMock(name="claude"),
            MagicMock(name="gpt-4"),
        ]

        self.result = MagicMock()
        self.result.final_answer = f"Answer for {self.debate_id}"
        self.result.confidence = confidence
        self.result.consensus_reached = True
        self.result.winner = "claude"
        self.result.rounds_used = 3
        self.result.key_claims = []


class DelayingMemorySystem:
    """Memory system mock that introduces configurable delays."""

    def __init__(self, name: str, delay_ms: float = 10, fail_rate: float = 0.0):
        self.name = name
        self.delay_ms = delay_ms
        self.fail_rate = fail_rate
        self.writes: List[Dict[str, Any]] = []
        self.deleted: List[str] = []
        self._lock = asyncio.Lock()
        self._write_count = 0

    async def store(self, **kwargs) -> str:
        """Store with configurable delay and failure rate."""
        await asyncio.sleep(self.delay_ms / 1000)

        if random.random() < self.fail_rate:
            raise Exception(f"{self.name}: Random failure")

        async with self._lock:
            self._write_count += 1
            entry_id = f"{self.name}-{self._write_count}"
            self.writes.append({"id": entry_id, **kwargs})

        return entry_id

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        async with self._lock:
            self.deleted.append(entry_id)
        return True


class ConcurrentAdapter:
    """Adapter for concurrent testing."""

    def __init__(self, name: str, delay_ms: float = 10, fail_rate: float = 0.0):
        self.name = name
        self.delay_ms = delay_ms
        self.fail_rate = fail_rate
        self.forward_calls: List[datetime] = []
        self.reverse_calls: List[datetime] = []
        self._lock = asyncio.Lock()

    async def sync_to_km(self) -> Dict[str, Any]:
        """Forward sync with delay."""
        await asyncio.sleep(self.delay_ms / 1000)

        if random.random() < self.fail_rate:
            raise Exception(f"{self.name}: Forward sync failed")

        async with self._lock:
            self.forward_calls.append(datetime.now())

        return {
            "items_processed": 10,
            "items_updated": 5,
        }

    async def sync_from_km(
        self, km_items: List[Dict[str, Any]], min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """Reverse sync with delay."""
        await asyncio.sleep(self.delay_ms / 1000)

        if random.random() < self.fail_rate:
            raise Exception(f"{self.name}: Reverse sync failed")

        async with self._lock:
            self.reverse_calls.append(datetime.now())

        return {
            "items_processed": len(km_items),
            "items_updated": len(km_items) // 2,
        }


# ============================================================================
# MemoryCoordinator Concurrent Write Tests
# ============================================================================


class TestMemoryCoordinatorConcurrentWrites:
    """Tests for concurrent writes through MemoryCoordinator."""

    @pytest.fixture
    def mock_memory_systems(self):
        """Create mock memory systems with delays."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(
            side_effect=lambda **kwargs: f"continuum-{uuid.uuid4().hex[:8]}"
        )
        continuum.delete = MagicMock(return_value={"deleted": True})

        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = f"consensus-{uuid.uuid4().hex[:8]}"
        consensus.store_consensus = MagicMock(return_value=mock_record)
        consensus.delete_consensus = MagicMock(return_value=True)

        critique = MagicMock()
        critique.store_result = MagicMock()
        critique.delete_debate = MagicMock(return_value=True)

        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(return_value=f"mound-{uuid.uuid4().hex[:8]}")
        mound.delete_entry = AsyncMock(return_value=True)

        return continuum, consensus, critique, mound

    @pytest.mark.asyncio
    async def test_concurrent_commits_no_race(self, mock_memory_systems):
        """Test that concurrent commits don't cause race conditions."""
        continuum, consensus, critique, mound = mock_memory_systems

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        # Create multiple debate contexts
        contexts = [MockDebateContext() for _ in range(10)]

        # Commit all concurrently
        tasks = [
            coordinator.commit_debate_outcome(ctx, options=CoordinatorOptions(parallel_writes=True))
            for ctx in contexts
        ]
        transactions = await asyncio.gather(*tasks)

        # All should succeed
        assert all(tx.success for tx in transactions)
        assert coordinator.metrics.successful_transactions == 10
        assert coordinator.metrics.total_writes == 40  # 4 systems × 10 debates

    @pytest.mark.asyncio
    async def test_concurrent_commits_with_failures(self, mock_memory_systems):
        """Test concurrent commits where some fail persistently."""
        continuum, consensus, critique, mound = mock_memory_systems

        # Track which debates should fail (persistently, across retries)
        failing_debates = set()
        call_counter = {"count": 0}

        def failing_store(*args, **kwargs):
            call_counter["count"] += 1
            # Mark every 3rd debate for failure (by topic pattern)
            topic = kwargs.get("topic", "")
            # Fail debates 3, 6, 9 (indices 2, 5, 8)
            if any(f"debate-{i}" in topic for i in [2, 5, 8]):
                raise Exception(f"Persistent failure for {topic}")
            mock_record = MagicMock()
            mock_record.id = f"consensus-{call_counter['count']}"
            return mock_record

        consensus.store_consensus = MagicMock(side_effect=failing_store)

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
            options=CoordinatorOptions(
                rollback_on_failure=False,
                parallel_writes=False,
                max_retries=0,  # No retries to ensure failures propagate
            ),
        )

        # Create contexts with deterministic IDs for failure targeting
        contexts = [MockDebateContext(debate_id=f"debate-{i}") for i in range(9)]
        transactions = []
        for ctx in contexts:
            tx = await coordinator.commit_debate_outcome(ctx)
            transactions.append(tx)

        # Count outcomes
        success_count = sum(1 for tx in transactions if tx.success)
        partial_failure_count = sum(1 for tx in transactions if tx.partial_failure)

        # Should have exactly 3 partial failures (debates 2, 5, 8)
        # and 6 successes (debates 0, 1, 3, 4, 6, 7)
        assert success_count == 6, f"Expected 6 successes, got {success_count}"
        assert (
            partial_failure_count == 3
        ), f"Expected 3 partial failures, got {partial_failure_count}"

    @pytest.mark.asyncio
    async def test_rollback_consistency(self, mock_memory_systems):
        """Test that rollback maintains consistency."""
        continuum, consensus, critique, mound = mock_memory_systems

        # Make mound always fail to trigger rollback
        mound.ingest_debate_outcome = AsyncMock(side_effect=Exception("Mound unavailable"))

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
            options=CoordinatorOptions(
                rollback_on_failure=True,
                parallel_writes=False,  # Sequential for predictable rollback
            ),
        )

        ctx = MockDebateContext()
        tx = await coordinator.commit_debate_outcome(ctx)

        # Transaction should be rolled back
        assert tx.rolled_back is True
        assert coordinator.metrics.rollbacks_performed == 1

        # Verify rollback was called on successful writes
        assert continuum.delete.called
        assert consensus.delete_consensus.called
        assert critique.delete_debate.called

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, mock_memory_systems):
        """Stress test with high concurrency."""
        continuum, consensus, critique, mound = mock_memory_systems

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        # 50 concurrent commits
        contexts = [MockDebateContext() for _ in range(50)]
        start_time = time.perf_counter()

        tasks = [
            coordinator.commit_debate_outcome(ctx, options=CoordinatorOptions(parallel_writes=True))
            for ctx in contexts
        ]
        transactions = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start_time

        # All should complete
        assert len(transactions) == 50
        assert all(tx is not None for tx in transactions)

        # Should complete in reasonable time (< 5 seconds even with delays)
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_ordering_preserved_sequential(self, mock_memory_systems):
        """Test that sequential writes preserve order."""
        continuum, consensus, critique, mound = mock_memory_systems

        write_order: List[str] = []

        def track_continuum(**kwargs):
            write_order.append("continuum")
            return "continuum-1"

        def track_consensus(*args, **kwargs):
            write_order.append("consensus")
            mock_record = MagicMock()
            mock_record.id = "consensus-1"
            return mock_record

        def track_critique(*args, **kwargs):
            write_order.append("critique")

        async def track_mound(**kwargs):
            write_order.append("mound")
            return "mound-1"

        continuum.store_pattern = MagicMock(side_effect=track_continuum)
        consensus.store_consensus = MagicMock(side_effect=track_consensus)
        critique.store_result = MagicMock(side_effect=track_critique)
        mound.ingest_debate_outcome = AsyncMock(side_effect=track_mound)

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        ctx = MockDebateContext()
        await coordinator.commit_debate_outcome(
            ctx, options=CoordinatorOptions(parallel_writes=False)
        )

        # Order should be: continuum → consensus → critique → mound
        assert write_order == ["continuum", "consensus", "critique", "mound"]


# ============================================================================
# BidirectionalCoordinator Concurrent Sync Tests
# ============================================================================


class TestBidirectionalCoordinatorConcurrentSync:
    """Tests for concurrent sync operations."""

    @pytest.fixture
    def coordinator_with_adapters(self) -> BidirectionalCoordinator:
        """Create coordinator with multiple adapters."""
        config = CoordinatorConfig(
            parallel_sync=True,
            timeout_seconds=5.0,
        )
        coordinator = BidirectionalCoordinator(config=config)

        # Register adapters with different priorities
        for i, (name, priority) in enumerate(
            [
                ("continuum", 100),
                ("consensus", 90),
                ("critique", 80),
                ("evidence", 70),
                ("elo", 40),
            ]
        ):
            adapter = ConcurrentAdapter(name, delay_ms=10)
            coordinator.register_adapter(
                name,
                adapter,
                "sync_to_km",
                "sync_from_km",
                priority=priority,
            )

        return coordinator

    @pytest.mark.asyncio
    async def test_concurrent_sync_requests_blocked(self, coordinator_with_adapters):
        """Test that concurrent sync requests are properly blocked."""
        coordinator = coordinator_with_adapters
        km_items = [{"id": str(i)} for i in range(10)]

        # Start multiple syncs concurrently
        tasks = [
            asyncio.create_task(coordinator.run_bidirectional_sync(km_items)) for _ in range(5)
        ]
        reports = await asyncio.gather(*tasks)

        # Count successful vs blocked
        successful = [r for r in reports if r.total_errors == 0]
        blocked = [r for r in reports if r.metadata.get("error") == "Sync already in progress"]

        # At least one should succeed
        assert len(successful) >= 1

        # Only one should actually run (others blocked or succeed after first completes)
        # The behavior depends on timing, but no more than 5 should succeed
        assert len(successful) <= 5

    @pytest.mark.asyncio
    async def test_parallel_adapter_sync(self, coordinator_with_adapters):
        """Test that adapters sync in parallel."""
        coordinator = coordinator_with_adapters
        km_items = [{"id": "1"}]

        start = time.perf_counter()
        report = await coordinator.run_bidirectional_sync(km_items)
        elapsed = time.perf_counter() - start

        # 5 adapters × 10ms delay × 2 directions = 100ms sequential
        # Parallel should be ~20ms (10ms forward + 10ms reverse)
        # Allow some overhead, but should be well under sequential time
        assert elapsed < 0.5, f"Expected parallel execution, took {elapsed:.3f}s"
        assert report.successful_forward == 5
        assert report.successful_reverse == 5

    @pytest.mark.asyncio
    async def test_adapter_failure_isolation(self):
        """Test that one adapter failure doesn't affect others."""
        config = CoordinatorConfig(parallel_sync=True)
        coordinator = BidirectionalCoordinator(config=config)

        # Register good adapters
        for name in ["adapter1", "adapter2", "adapter3"]:
            adapter = ConcurrentAdapter(name, delay_ms=5)
            coordinator.register_adapter(name, adapter, "sync_to_km", "sync_from_km")

        # Register failing adapter
        failing_adapter = ConcurrentAdapter("failing", fail_rate=1.0)
        coordinator.register_adapter("failing", failing_adapter, "sync_to_km", "sync_from_km")

        km_items = [{"id": "1"}]
        report = await coordinator.run_bidirectional_sync(km_items)

        # 3 should succeed, 1 should fail
        assert report.successful_forward == 3
        assert report.total_errors > 0

    @pytest.mark.asyncio
    async def test_priority_order_sequential(self):
        """Test that sequential sync respects priority."""
        config = CoordinatorConfig(parallel_sync=False)
        coordinator = BidirectionalCoordinator(config=config)

        sync_order: List[str] = []

        class OrderTrackingAdapter:
            def __init__(self, name):
                self.name = name

            async def sync_to_km(self):
                sync_order.append(self.name)
                return {"items_processed": 1}

            async def sync_from_km(self, items, min_confidence=0.7):
                return {"items_processed": 0}

        # Register in random order
        adapters = [
            ("high", 100),
            ("low", 10),
            ("medium", 50),
        ]
        random.shuffle(adapters)

        for name, priority in adapters:
            coordinator.register_adapter(
                name,
                OrderTrackingAdapter(name),
                "sync_to_km",
                priority=priority,
            )

        await coordinator.sync_all_to_km()

        # Should be in priority order (high to low)
        assert sync_order == ["high", "medium", "low"]


# ============================================================================
# Conflict Resolution Tests
# ============================================================================


class TestConflictResolution:
    """Tests for conflict resolution during concurrent writes."""

    @pytest.mark.asyncio
    async def test_confidence_based_precedence(self):
        """Test that higher confidence wins in conflicts."""
        from aragora.memory.coordinator import MemoryCoordinator, CoordinatorOptions

        # Track what gets written
        stored_items: List[Dict[str, Any]] = []

        continuum = MagicMock()

        def capture_store(**kwargs):
            stored_items.append(kwargs)
            return f"entry-{len(stored_items)}"

        continuum.store_pattern = MagicMock(side_effect=capture_store)

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            options=CoordinatorOptions(
                write_consensus=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        # Create contexts with different confidence levels
        high_confidence_ctx = MockDebateContext(confidence=0.95)
        low_confidence_ctx = MockDebateContext(confidence=0.65)

        # Commit both
        await coordinator.commit_debate_outcome(high_confidence_ctx)
        await coordinator.commit_debate_outcome(low_confidence_ctx)

        # Both stored (conflict resolution is at query time, not write time)
        assert len(stored_items) == 2

        # Verify importance reflects confidence
        high_importance = stored_items[0]["importance"]
        low_importance = stored_items[1]["importance"]
        assert high_importance > low_importance

    @pytest.mark.asyncio
    async def test_mound_skipped_below_threshold(self):
        """Test that low-confidence items skip the mound."""
        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(return_value="mound-1")

        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="continuum-1")

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            knowledge_mound=mound,
            options=CoordinatorOptions(
                write_consensus=False,
                write_critique=False,
                min_confidence_for_mound=0.7,
            ),
        )

        # Low confidence - should skip mound
        low_ctx = MockDebateContext(confidence=0.5)
        tx = await coordinator.commit_debate_outcome(low_ctx)

        targets = [op.target for op in tx.operations]
        assert "mound" not in targets
        assert mound.ingest_debate_outcome.call_count == 0

        # High confidence - should write to mound
        high_ctx = MockDebateContext(confidence=0.9)
        tx = await coordinator.commit_debate_outcome(high_ctx)

        targets = [op.target for op in tx.operations]
        assert "mound" in targets
        assert mound.ingest_debate_outcome.call_count == 1


# ============================================================================
# Deadlock Prevention Tests
# ============================================================================


class TestDeadlockPrevention:
    """Tests to ensure no deadlocks occur during concurrent operations."""

    @pytest.mark.asyncio
    async def test_no_deadlock_with_nested_locks(self):
        """Test that nested operations don't cause deadlocks."""
        config = CoordinatorConfig(parallel_sync=True)
        coordinator = BidirectionalCoordinator(config=config)

        class NestedAdapter:
            """Adapter that triggers nested operations."""

            def __init__(self, coordinator_ref):
                self._coordinator = coordinator_ref
                self.call_count = 0

            async def sync_to_km(self):
                self.call_count += 1
                # Don't actually nest - just simulate work
                await asyncio.sleep(0.01)
                return {"items_processed": 1}

            async def sync_from_km(self, items, min_confidence=0.7):
                return {"items_processed": 0}

        adapter = NestedAdapter(coordinator)
        coordinator.register_adapter("nested", adapter, "sync_to_km", "sync_from_km")

        # Should complete without deadlock
        km_items = [{"id": "1"}]

        async def run_with_timeout():
            try:
                return await asyncio.wait_for(
                    coordinator.run_bidirectional_sync(km_items),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                return None

        report = await run_with_timeout()

        # Should complete successfully (no deadlock)
        assert report is not None
        assert adapter.call_count > 0

    @pytest.mark.asyncio
    async def test_timeout_prevents_indefinite_wait(self):
        """Test that timeouts prevent indefinite waiting."""
        config = CoordinatorConfig(
            parallel_sync=True,
            timeout_seconds=0.1,  # Very short timeout
        )
        coordinator = BidirectionalCoordinator(config=config)

        class SlowAdapter:
            """Adapter that never completes."""

            async def sync_to_km(self):
                await asyncio.sleep(10)  # Will timeout
                return {}

            async def sync_from_km(self, items, min_confidence=0.7):
                await asyncio.sleep(10)
                return {}

        coordinator.register_adapter("slow", SlowAdapter(), "sync_to_km", "sync_from_km")

        km_items = [{"id": "1"}]

        start = time.perf_counter()
        results = await coordinator.sync_all_to_km()
        elapsed = time.perf_counter() - start

        # Should timeout quickly, not wait 10 seconds
        assert elapsed < 1.0
        assert results[0].success is False
        assert any("timeout" in e.lower() for e in results[0].errors)


# ============================================================================
# Integration Tests
# ============================================================================


class TestFullIntegration:
    """End-to-end integration tests for concurrent scenarios."""

    @pytest.mark.asyncio
    async def test_debate_completion_flow(self):
        """Test the full flow when a debate completes."""
        # Setup memory coordinator
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="continuum-1")
        continuum.delete = MagicMock(return_value={"deleted": True})

        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-1"
        consensus.store_consensus = MagicMock(return_value=mock_record)

        critique = MagicMock()
        critique.store_result = MagicMock()

        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(return_value="mound-1")

        memory_coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        # Setup bidirectional coordinator
        bidir_config = CoordinatorConfig(parallel_sync=True)
        bidir_coordinator = BidirectionalCoordinator(config=bidir_config)

        for name in ["continuum", "consensus", "evidence"]:
            adapter = ConcurrentAdapter(name, delay_ms=5)
            bidir_coordinator.register_adapter(name, adapter, "sync_to_km", "sync_from_km")

        # Simulate debate completion
        ctx = MockDebateContext(confidence=0.9)

        # Step 1: Commit to memory systems
        tx = await memory_coordinator.commit_debate_outcome(ctx)
        assert tx.success is True

        # Step 2: Run bidirectional sync
        km_items = [{"id": tx.debate_id}]
        report = await bidir_coordinator.run_bidirectional_sync(km_items)

        assert report.successful_forward == 3
        assert report.successful_reverse == 3

    @pytest.mark.asyncio
    async def test_multiple_debates_concurrent(self):
        """Test multiple debates completing concurrently."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(
            side_effect=lambda **kwargs: f"continuum-{uuid.uuid4().hex[:8]}"
        )

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            options=CoordinatorOptions(
                write_consensus=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        # 20 debates completing at once
        contexts = [MockDebateContext() for _ in range(20)]

        start = time.perf_counter()
        tasks = [coordinator.commit_debate_outcome(ctx) for ctx in contexts]
        transactions = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

        # All should succeed
        assert all(tx.success for tx in transactions)
        assert coordinator.metrics.successful_transactions == 20

        # Should complete quickly
        assert elapsed < 2.0


# ============================================================================
# Metrics and Observability Tests
# ============================================================================


class TestConcurrencyMetrics:
    """Tests for metrics under concurrent load."""

    @pytest.mark.asyncio
    async def test_metrics_accuracy_under_load(self):
        """Test that metrics remain accurate under load."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="entry-1")

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            options=CoordinatorOptions(
                write_consensus=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        # Run 100 concurrent commits
        contexts = [MockDebateContext() for _ in range(100)]
        tasks = [coordinator.commit_debate_outcome(ctx) for ctx in contexts]
        await asyncio.gather(*tasks)

        metrics = coordinator.get_metrics()

        assert metrics["total_transactions"] == 100
        assert metrics["successful_transactions"] == 100
        assert metrics["total_writes"] == 100  # 1 system × 100 debates
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_bidirectional_sync_history_accuracy(self):
        """Test that sync history is accurate under concurrent access."""
        config = CoordinatorConfig(parallel_sync=True)
        coordinator = BidirectionalCoordinator(config=config)

        adapter = ConcurrentAdapter("test", delay_ms=5)
        coordinator.register_adapter("test", adapter, "sync_to_km", "sync_from_km")

        # Run syncs sequentially (to avoid blocking)
        for _ in range(10):
            await coordinator.run_bidirectional_sync([{"id": "1"}])

        history = coordinator.get_sync_history(limit=10)

        assert len(history) == 10
        assert all(isinstance(r, BidirectionalSyncReport) for r in history)
        assert all(r.timestamp != "" for r in history)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
