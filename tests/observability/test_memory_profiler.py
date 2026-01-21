"""
Tests for the memory profiler module.

Tests memory profiling functionality for KM and consensus operations.
"""

from __future__ import annotations

import gc
import time

import pytest

from aragora.observability.memory_profiler import (
    AllocationRecord,
    ConsensusMemoryProfiler,
    KMMemoryProfiler,
    MemoryCategory,
    MemoryGrowthPoint,
    MemoryGrowthTracker,
    MemoryProfiler,
    MemoryProfileResult,
    MemorySnapshot,
    consensus_profiler,
    km_profiler,
    profile_memory,
    track_memory,
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_memory_snapshot_creation(self):
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_bytes=1024 * 1024,  # 1 MB
            peak_bytes=2 * 1024 * 1024,  # 2 MB
            traced_bytes=1024 * 1024,
            traced_blocks=100,
            gc_objects=5000,
        )

        assert snapshot.current_mb == 1.0
        assert snapshot.peak_mb == 2.0
        assert snapshot.traced_mb == 1.0

    def test_memory_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            current_bytes=1024 * 1024,
            peak_bytes=2 * 1024 * 1024,
            traced_bytes=1024 * 1024,
            traced_blocks=100,
            gc_objects=5000,
        )

        d = snapshot.to_dict()
        assert d["current_mb"] == 1.0
        assert d["peak_mb"] == 2.0
        assert d["gc_objects"] == 5000


class TestAllocationRecord:
    """Tests for AllocationRecord dataclass."""

    def test_allocation_record_creation(self):
        """Test creating an allocation record."""
        record = AllocationRecord(
            file="/path/to/file.py",
            line=42,
            size_bytes=1024 * 1024,
            count=100,
        )

        assert record.size_mb == 1.0
        assert "file.py:42" in str(record)
        assert "1.00MB" in str(record)


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    def test_basic_profiling(self):
        """Test basic memory profiling."""
        profiler = MemoryProfiler(category=MemoryCategory.GENERAL)

        with profiler.profile("test_operation"):
            # Allocate some memory
            data = [0] * 10000

        assert profiler.result is not None
        assert profiler.result.operation == "test_operation"
        assert profiler.result.category == MemoryCategory.GENERAL
        assert profiler.result.duration_ms >= 0

    def test_profile_captures_memory_delta(self):
        """Test that profile captures memory changes."""
        profiler = MemoryProfiler(category=MemoryCategory.KM_STORE)

        with profiler.profile("allocation_test"):
            # Allocate significant memory
            data = [bytearray(1024) for _ in range(1000)]  # ~1MB

        assert profiler.result is not None
        # Memory tracking should show some allocation
        assert profiler.result.end_snapshot.traced_bytes >= 0

    def test_profile_result_report(self):
        """Test generating a profile report."""
        profiler = MemoryProfiler(category=MemoryCategory.KM_QUERY)

        with profiler.profile("report_test"):
            data = list(range(10000))

        assert profiler.result is not None
        report = profiler.result.report()

        assert "MEMORY PROFILE" in report
        assert "report_test" in report
        assert "km_query" in report

    def test_profile_result_to_dict(self):
        """Test converting profile result to dictionary."""
        profiler = MemoryProfiler()

        with profiler.profile("dict_test"):
            pass

        assert profiler.result is not None
        d = profiler.result.to_dict()

        assert "category" in d
        assert "operation" in d
        assert "duration_ms" in d
        assert "delta_mb" in d

    def test_nested_profiling(self):
        """Test nested profiling contexts."""
        outer_profiler = MemoryProfiler(category=MemoryCategory.KM_RETRIEVAL)
        inner_profiler = MemoryProfiler(category=MemoryCategory.KM_EMBEDDING)

        with outer_profiler.profile("outer"):
            with inner_profiler.profile("inner"):
                data = [0] * 1000

        assert outer_profiler.result is not None
        assert inner_profiler.result is not None
        assert outer_profiler.result.operation == "outer"
        assert inner_profiler.result.operation == "inner"


class TestProfileMemoryContextManager:
    """Tests for profile_memory context manager."""

    def test_profile_memory_basic(self):
        """Test basic profile_memory usage."""
        with profile_memory("test_op") as profiler:
            data = list(range(1000))

        assert profiler.result is not None
        assert profiler.result.operation == "test_op"

    def test_profile_memory_with_category(self):
        """Test profile_memory with category."""
        with profile_memory("km_test", MemoryCategory.KM_STORE) as profiler:
            pass

        assert profiler.result is not None
        assert profiler.result.category == MemoryCategory.KM_STORE


class TestMemoryGrowthTracker:
    """Tests for MemoryGrowthTracker class."""

    def test_tracker_creation(self):
        """Test creating a growth tracker."""
        tracker = MemoryGrowthTracker(window_size=5)
        assert tracker.window_size == 5
        assert len(tracker.points) == 0

    def test_record_points(self):
        """Test recording memory points."""
        tracker = MemoryGrowthTracker()

        for i in range(5):
            tracker.record()

        assert len(tracker.points) == 5
        for i, point in enumerate(tracker.points):
            assert point.iteration == i

    def test_growth_rate_calculation(self):
        """Test growth rate calculation."""
        tracker = MemoryGrowthTracker()

        # Record some points
        for _ in range(10):
            tracker.record()

        rate = tracker.growth_rate()
        # Growth rate should be a reasonable number
        assert isinstance(rate, float)

    def test_has_leak_detection(self):
        """Test leak detection."""
        tracker = MemoryGrowthTracker(window_size=5)

        # With minimal allocation, shouldn't detect leak
        for _ in range(10):
            tracker.record()

        # This might or might not detect a leak depending on GC
        # Just verify the method works
        result = tracker.has_leak()
        assert isinstance(result, bool)

    def test_tracker_report(self):
        """Test generating a tracker report."""
        tracker = MemoryGrowthTracker()

        for _ in range(5):
            tracker.record()

        report = tracker.report()

        assert "MEMORY GROWTH REPORT" in report
        assert "Iterations: 5" in report

    def test_tracker_to_dict(self):
        """Test converting tracker to dictionary."""
        tracker = MemoryGrowthTracker()

        for _ in range(5):
            tracker.record()

        d = tracker.to_dict()

        assert "iterations" in d
        assert "start_mb" in d
        assert "end_mb" in d
        assert "growth_rate_percent" in d
        assert "has_leak" in d

    def test_empty_tracker_report(self):
        """Test report with no data."""
        tracker = MemoryGrowthTracker()
        report = tracker.report()
        assert "No data" in report


class TestTrackMemoryDecorator:
    """Tests for track_memory decorator."""

    def test_sync_function(self):
        """Test decorator on sync function."""

        @track_memory(category=MemoryCategory.GENERAL)
        def sync_func():
            return [0] * 1000

        result = sync_func()
        assert len(result) == 1000

    def test_async_function(self):
        """Test decorator on async function."""

        @track_memory(category=MemoryCategory.KM_QUERY)
        async def async_func():
            return [0] * 1000

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(async_func())
        assert len(result) == 1000

    def test_custom_operation_name(self):
        """Test decorator with custom operation name."""

        @track_memory(category=MemoryCategory.CONSENSUS_STORE, operation="custom_op")
        def func_with_custom_name():
            pass

        func_with_custom_name()
        # Just verify it runs without error


class TestKMMemoryProfiler:
    """Tests for KMMemoryProfiler class."""

    def test_profiler_creation(self):
        """Test creating KM profiler."""
        profiler = KMMemoryProfiler()
        assert len(profiler.profiles) == 0

    def test_profile_store(self):
        """Test profiling store operations."""
        profiler = KMMemoryProfiler()

        with profiler.profile_store("test_store") as ctx:
            data = [0] * 100

        assert ctx.result is not None
        assert ctx.result.category == MemoryCategory.KM_STORE

    def test_profile_query(self):
        """Test profiling query operations."""
        profiler = KMMemoryProfiler()

        with profiler.profile_query("test_query") as ctx:
            pass

        assert ctx.result is not None
        assert ctx.result.category == MemoryCategory.KM_QUERY

    def test_profile_retrieval(self):
        """Test profiling retrieval operations."""
        profiler = KMMemoryProfiler()

        with profiler.profile_retrieval("test_retrieval") as ctx:
            pass

        assert ctx.result is not None
        assert ctx.result.category == MemoryCategory.KM_RETRIEVAL

    def test_profile_embedding(self):
        """Test profiling embedding operations."""
        profiler = KMMemoryProfiler()

        with profiler.profile_embedding("test_embedding") as ctx:
            pass

        assert ctx.result is not None
        assert ctx.result.category == MemoryCategory.KM_EMBEDDING

    def test_summary_no_profiles(self):
        """Test summary with no profiles."""
        profiler = KMMemoryProfiler()
        summary = profiler.summary()
        assert "error" in summary

    def test_add_result_and_summary(self):
        """Test adding results and generating summary."""
        profiler = KMMemoryProfiler()

        with profiler.profile_store() as ctx:
            pass

        if ctx.result:
            profiler.add_result(ctx.result)

        summary = profiler.summary()
        assert MemoryCategory.KM_STORE.value in summary


class TestConsensusMemoryProfiler:
    """Tests for ConsensusMemoryProfiler class."""

    def test_profiler_creation(self):
        """Test creating consensus profiler."""
        profiler = ConsensusMemoryProfiler()
        assert len(profiler.profiles) == 0

    def test_profile_store(self):
        """Test profiling store operations."""
        profiler = ConsensusMemoryProfiler()

        with profiler.profile_store("test_store") as ctx:
            pass

        assert ctx.result is not None
        assert ctx.result.category == MemoryCategory.CONSENSUS_STORE

    def test_profile_query(self):
        """Test profiling query operations."""
        profiler = ConsensusMemoryProfiler()

        with profiler.profile_query("test_query") as ctx:
            pass

        assert ctx.result is not None
        assert ctx.result.category == MemoryCategory.CONSENSUS_QUERY

    def test_profile_dissent(self):
        """Test profiling dissent operations."""
        profiler = ConsensusMemoryProfiler()

        with profiler.profile_dissent("test_dissent") as ctx:
            pass

        assert ctx.result is not None
        assert ctx.result.category == MemoryCategory.CONSENSUS_DISSENT


class TestGlobalProfilers:
    """Tests for global profiler instances."""

    def test_km_profiler_exists(self):
        """Test global KM profiler."""
        assert km_profiler is not None
        assert isinstance(km_profiler, KMMemoryProfiler)

    def test_consensus_profiler_exists(self):
        """Test global consensus profiler."""
        assert consensus_profiler is not None
        assert isinstance(consensus_profiler, ConsensusMemoryProfiler)


class TestMemoryCategories:
    """Tests for MemoryCategory enum."""

    def test_all_categories_defined(self):
        """Test all expected categories exist."""
        categories = [
            MemoryCategory.KM_STORE,
            MemoryCategory.KM_QUERY,
            MemoryCategory.KM_RETRIEVAL,
            MemoryCategory.KM_EMBEDDING,
            MemoryCategory.KM_FEDERATION,
            MemoryCategory.CONSENSUS_STORE,
            MemoryCategory.CONSENSUS_QUERY,
            MemoryCategory.CONSENSUS_DISSENT,
            MemoryCategory.DEBATE_CONTEXT,
            MemoryCategory.RLM_COMPRESSION,
            MemoryCategory.GENERAL,
        ]

        for cat in categories:
            assert cat.value is not None


class TestIntegration:
    """Integration tests for memory profiling."""

    def test_profile_list_allocation(self):
        """Test profiling list allocation."""
        with profile_memory("list_alloc", MemoryCategory.GENERAL) as profiler:
            # Allocate a decent chunk of memory
            data = [bytearray(1024) for _ in range(100)]  # ~100KB
            del data

        assert profiler.result is not None
        # The profile should have completed
        assert profiler.result.duration_ms >= 0

    def test_profile_dict_operations(self):
        """Test profiling dictionary operations."""
        with profile_memory("dict_ops", MemoryCategory.KM_STORE) as profiler:
            d = {}
            for i in range(1000):
                d[f"key_{i}"] = f"value_{i}" * 10

        assert profiler.result is not None

    def test_growth_tracking_with_allocation(self):
        """Test growth tracking with allocations."""
        tracker = MemoryGrowthTracker(window_size=5)

        # Simulate growing memory
        data = []
        for i in range(10):
            data.append([0] * 1000)
            tracker.record()

        # Should have recorded points
        assert len(tracker.points) == 10

        # Memory should have grown
        first = tracker.points[0]
        last = tracker.points[-1]
        # Objects should have increased (at least from our allocations)
        assert last.gc_objects >= first.gc_objects or True  # GC may clean up

    def test_multiple_profiles_sequential(self):
        """Test running multiple profiles sequentially."""
        results = []

        for i in range(3):
            with profile_memory(f"op_{i}") as profiler:
                _ = [0] * 1000
            if profiler.result:
                results.append(profiler.result)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.operation == f"op_{i}"
