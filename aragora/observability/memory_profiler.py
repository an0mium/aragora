"""
Memory Profiling for Aragora.

Provides tools to track memory usage in Knowledge Mound and Consensus Store
operations, identifying leaks, growth patterns, and optimization opportunities.

Usage:
    from aragora.observability.memory_profiler import (
        MemoryProfiler,
        profile_memory,
        track_memory_growth,
    )

    # Profile a code block
    with profile_memory("km_query") as profiler:
        result = await km.query("test query")

    print(profiler.report())

    # Track memory growth over time
    tracker = MemoryGrowthTracker()
    for i in range(100):
        await process_batch()
        tracker.record()
    print(tracker.report())
"""

from __future__ import annotations

import gc
import logging
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Memory thresholds
MEMORY_WARNING_MB = 100  # Warn if operation uses > 100MB
MEMORY_CRITICAL_MB = 500  # Critical if operation uses > 500MB
GROWTH_RATE_WARNING = 0.10  # Warn if memory grows > 10% per iteration


class MemoryCategory(str, Enum):
    """Categories of memory-tracked operations."""

    KM_STORE = "km_store"
    KM_QUERY = "km_query"
    KM_RETRIEVAL = "km_retrieval"
    KM_EMBEDDING = "km_embedding"
    KM_FEDERATION = "km_federation"
    CONSENSUS_STORE = "consensus_store"
    CONSENSUS_QUERY = "consensus_query"
    CONSENSUS_DISSENT = "consensus_dissent"
    DEBATE_CONTEXT = "debate_context"
    RLM_COMPRESSION = "rlm_compression"
    GENERAL = "general"


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""

    timestamp: float
    current_bytes: int
    peak_bytes: int
    traced_bytes: int
    traced_blocks: int
    gc_objects: int

    @property
    def current_mb(self) -> float:
        return self.current_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    @property
    def traced_mb(self) -> float:
        return self.traced_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_mb": round(self.current_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "traced_mb": round(self.traced_mb, 2),
            "traced_blocks": self.traced_blocks,
            "gc_objects": self.gc_objects,
        }


@dataclass
class AllocationRecord:
    """Record of a memory allocation hotspot."""

    file: str
    line: int
    size_bytes: int
    count: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def __str__(self) -> str:
        return f"{self.file}:{self.line} - {self.size_mb:.2f}MB ({self.count} blocks)"


@dataclass
class MemoryProfileResult:
    """Result of profiling a memory operation."""

    category: MemoryCategory
    operation: str
    start_snapshot: MemorySnapshot
    end_snapshot: MemorySnapshot
    duration_ms: float
    top_allocations: list[AllocationRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def delta_bytes(self) -> int:
        return self.end_snapshot.current_bytes - self.start_snapshot.current_bytes

    @property
    def delta_mb(self) -> float:
        return self.delta_bytes / (1024 * 1024)

    @property
    def peak_delta_bytes(self) -> int:
        return self.end_snapshot.peak_bytes - self.start_snapshot.current_bytes

    @property
    def peak_delta_mb(self) -> float:
        return self.peak_delta_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "delta_mb": round(self.delta_mb, 2),
            "peak_delta_mb": round(self.peak_delta_mb, 2),
            "start": self.start_snapshot.to_dict(),
            "end": self.end_snapshot.to_dict(),
            "top_allocations": [
                {"file": a.file, "line": a.line, "size_mb": a.size_mb, "count": a.count}
                for a in self.top_allocations[:5]
            ],
            "warnings": self.warnings,
        }

    def report(self, verbose: bool = False) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 60,
            f"MEMORY PROFILE: {self.operation}",
            f"Category: {self.category.value}",
            "=" * 60,
            f"Duration: {self.duration_ms:.2f}ms",
            f"Memory delta: {self.delta_mb:+.2f}MB",
            f"Peak usage: {self.peak_delta_mb:.2f}MB",
            f"GC objects delta: {self.end_snapshot.gc_objects - self.start_snapshot.gc_objects:+d}",
            "",
        ]

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        if self.top_allocations:
            lines.append("TOP ALLOCATIONS:")
            lines.append("-" * 40)
            for alloc in self.top_allocations[: 10 if verbose else 5]:
                lines.append(f"  {alloc}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class MemoryProfiler:
    """
    Profiles memory usage for operations.

    Uses tracemalloc for detailed allocation tracking and gc for object counting.

    Usage:
        profiler = MemoryProfiler(category=MemoryCategory.KM_QUERY)
        with profiler.profile("semantic_search"):
            result = await km.query(...)
        print(profiler.result.report())
    """

    def __init__(self, category: MemoryCategory = MemoryCategory.GENERAL):
        self.category = category
        self.result: Optional[MemoryProfileResult] = None
        self._tracing_started_by_us = False

    def _take_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory state."""
        gc.collect()

        traced_bytes = 0
        traced_blocks = 0
        if tracemalloc.is_tracing():
            traced_bytes, traced_blocks = tracemalloc.get_traced_memory()

        return MemorySnapshot(
            timestamp=time.time(),
            current_bytes=traced_bytes,
            peak_bytes=traced_bytes,  # Will be updated at end
            traced_bytes=traced_bytes,
            traced_blocks=traced_blocks,
            gc_objects=len(gc.get_objects()),
        )

    def _get_top_allocations(self, limit: int = 10) -> list[AllocationRecord]:
        """Get top memory allocation hotspots."""
        if not tracemalloc.is_tracing():
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        records = []
        for stat in top_stats[:limit]:
            frame = stat.traceback[0] if stat.traceback else None
            if frame:
                records.append(
                    AllocationRecord(
                        file=frame.filename,
                        line=frame.lineno,
                        size_bytes=stat.size,
                        count=stat.count,
                    )
                )

        return records

    @contextmanager
    def profile(self, operation: str):
        """Context manager for profiling an operation."""
        # Start tracing if not already
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracing_started_by_us = True

        tracemalloc.reset_peak()
        start_snapshot = self._take_snapshot()
        start_time = time.perf_counter()

        try:
            yield self
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Get peak memory
            current, peak = tracemalloc.get_traced_memory()
            gc.collect()

            end_snapshot = MemorySnapshot(
                timestamp=time.time(),
                current_bytes=current,
                peak_bytes=peak,
                traced_bytes=current,
                traced_blocks=len(gc.get_objects()),
                gc_objects=len(gc.get_objects()),
            )

            top_allocations = self._get_top_allocations()

            # Generate warnings
            warnings = []
            peak_mb = (peak - start_snapshot.current_bytes) / (1024 * 1024)
            if peak_mb > MEMORY_CRITICAL_MB:
                warnings.append(
                    f"CRITICAL: Peak memory usage {peak_mb:.1f}MB exceeds {MEMORY_CRITICAL_MB}MB threshold"
                )
            elif peak_mb > MEMORY_WARNING_MB:
                warnings.append(
                    f"WARNING: Peak memory usage {peak_mb:.1f}MB exceeds {MEMORY_WARNING_MB}MB threshold"
                )

            self.result = MemoryProfileResult(
                category=self.category,
                operation=operation,
                start_snapshot=start_snapshot,
                end_snapshot=end_snapshot,
                duration_ms=duration_ms,
                top_allocations=top_allocations,
                warnings=warnings,
            )

            if warnings:
                for w in warnings:
                    logger.warning(f"Memory profile [{operation}]: {w}")

            # Stop tracing if we started it
            if self._tracing_started_by_us:
                tracemalloc.stop()
                self._tracing_started_by_us = False


@contextmanager
def profile_memory(
    operation: str,
    category: MemoryCategory = MemoryCategory.GENERAL,
) -> MemoryProfiler:
    """Convenience context manager for memory profiling."""
    profiler = MemoryProfiler(category=category)
    with profiler.profile(operation):
        yield profiler


@dataclass
class MemoryGrowthPoint:
    """A single point in memory growth tracking."""

    iteration: int
    timestamp: float
    memory_bytes: int
    gc_objects: int

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)


class MemoryGrowthTracker:
    """
    Tracks memory growth over iterations to detect leaks.

    Usage:
        tracker = MemoryGrowthTracker()
        for i in range(100):
            process_item(i)
            tracker.record()

        if tracker.has_leak():
            print(tracker.report())
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.points: list[MemoryGrowthPoint] = []
        self._start_tracing()

    def _start_tracing(self) -> None:
        """Ensure tracemalloc is running."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def record(self) -> None:
        """Record current memory state."""
        gc.collect()
        current, _ = tracemalloc.get_traced_memory()

        self.points.append(
            MemoryGrowthPoint(
                iteration=len(self.points),
                timestamp=time.time(),
                memory_bytes=current,
                gc_objects=len(gc.get_objects()),
            )
        )

    def growth_rate(self) -> float:
        """Calculate average growth rate per iteration."""
        if len(self.points) < 2:
            return 0.0

        first = self.points[0]
        last = self.points[-1]

        if first.memory_bytes == 0:
            return 0.0

        total_growth = (last.memory_bytes - first.memory_bytes) / first.memory_bytes
        return total_growth / (len(self.points) - 1)

    def has_leak(self) -> bool:
        """Check if memory growth suggests a leak."""
        if len(self.points) < self.window_size:
            return False

        # Check if memory consistently grows
        recent = self.points[-self.window_size :]
        growth_count = sum(
            1 for i in range(1, len(recent)) if recent[i].memory_bytes > recent[i - 1].memory_bytes
        )

        # If >80% of recent iterations show growth, likely leak
        return growth_count > self.window_size * 0.8

    def report(self) -> str:
        """Generate growth report."""
        if not self.points:
            return "No data recorded"

        first = self.points[0]
        last = self.points[-1]

        lines = [
            "=" * 60,
            "MEMORY GROWTH REPORT",
            "=" * 60,
            f"Iterations: {len(self.points)}",
            f"Duration: {last.timestamp - first.timestamp:.2f}s",
            f"Start memory: {first.memory_mb:.2f}MB",
            f"End memory: {last.memory_mb:.2f}MB",
            f"Delta: {last.memory_mb - first.memory_mb:+.2f}MB",
            f"Growth rate: {self.growth_rate() * 100:.2f}% per iteration",
            f"GC objects delta: {last.gc_objects - first.gc_objects:+d}",
            "",
        ]

        if self.has_leak():
            lines.append("WARNING: Potential memory leak detected!")
            lines.append("")

        # Show growth trend
        if len(self.points) >= 5:
            lines.append("GROWTH TREND (sampled):")
            lines.append("-" * 40)
            step = max(1, len(self.points) // 5)
            for point in self.points[::step][:5]:
                lines.append(
                    f"  [{point.iteration:4d}] {point.memory_mb:8.2f}MB  ({point.gc_objects} objects)"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        if not self.points:
            return {"error": "No data"}

        first = self.points[0]
        last = self.points[-1]

        return {
            "iterations": len(self.points),
            "start_mb": round(first.memory_mb, 2),
            "end_mb": round(last.memory_mb, 2),
            "delta_mb": round(last.memory_mb - first.memory_mb, 2),
            "growth_rate_percent": round(self.growth_rate() * 100, 2),
            "has_leak": self.has_leak(),
        }


# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


def track_memory(
    category: MemoryCategory = MemoryCategory.GENERAL,
    operation: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to track memory usage of a function.

    Usage:
        @track_memory(category=MemoryCategory.KM_QUERY)
        async def query_knowledge(query: str):
            ...
    """

    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with profile_memory(op_name, category) as profiler:
                result = func(*args, **kwargs)
                if profiler.result and profiler.result.delta_mb > MEMORY_WARNING_MB:
                    logger.warning(
                        f"High memory usage in {op_name}: {profiler.result.delta_mb:.1f}MB"
                    )
                return result

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with profile_memory(op_name, category) as profiler:
                result = await func(*args, **kwargs)
                if profiler.result and profiler.result.delta_mb > MEMORY_WARNING_MB:
                    logger.warning(
                        f"High memory usage in {op_name}: {profiler.result.delta_mb:.1f}MB"
                    )
                return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    return decorator


class KMMemoryProfiler:
    """
    Specialized profiler for Knowledge Mound operations.

    Tracks:
    - Node storage memory
    - Query memory (embedding generation + vector search)
    - Retrieval memory
    - Federation sync memory
    """

    def __init__(self):
        self.profiles: list[MemoryProfileResult] = []

    def profile_store(self, operation: str = "km_store"):
        """Profile a KM store operation."""
        return profile_memory(operation, MemoryCategory.KM_STORE)

    def profile_query(self, operation: str = "km_query"):
        """Profile a KM query operation."""
        return profile_memory(operation, MemoryCategory.KM_QUERY)

    def profile_retrieval(self, operation: str = "km_retrieval"):
        """Profile a KM retrieval operation."""
        return profile_memory(operation, MemoryCategory.KM_RETRIEVAL)

    def profile_embedding(self, operation: str = "km_embedding"):
        """Profile embedding generation."""
        return profile_memory(operation, MemoryCategory.KM_EMBEDDING)

    def add_result(self, result: MemoryProfileResult) -> None:
        """Add a profiling result."""
        self.profiles.append(result)

    def summary(self) -> dict[str, Any]:
        """Generate summary of all profiles."""
        if not self.profiles:
            return {"error": "No profiles recorded"}

        by_category: dict[str, list[MemoryProfileResult]] = {}
        for p in self.profiles:
            cat = p.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(p)

        summary = {}
        for cat, profiles in by_category.items():
            deltas = [p.delta_mb for p in profiles]
            peaks = [p.peak_delta_mb for p in profiles]
            durations = [p.duration_ms for p in profiles]

            summary[cat] = {
                "count": len(profiles),
                "avg_delta_mb": round(sum(deltas) / len(deltas), 2),
                "max_delta_mb": round(max(deltas), 2),
                "avg_peak_mb": round(sum(peaks) / len(peaks), 2),
                "max_peak_mb": round(max(peaks), 2),
                "avg_duration_ms": round(sum(durations) / len(durations), 2),
                "total_warnings": sum(len(p.warnings) for p in profiles),
            }

        return summary


class ConsensusMemoryProfiler:
    """
    Specialized profiler for Consensus Store operations.

    Tracks:
    - Consensus storage memory
    - Dissent retrieval memory
    - Cache behavior
    """

    def __init__(self):
        self.profiles: list[MemoryProfileResult] = []

    def profile_store(self, operation: str = "consensus_store"):
        """Profile a consensus store operation."""
        return profile_memory(operation, MemoryCategory.CONSENSUS_STORE)

    def profile_query(self, operation: str = "consensus_query"):
        """Profile a consensus query operation."""
        return profile_memory(operation, MemoryCategory.CONSENSUS_QUERY)

    def profile_dissent(self, operation: str = "consensus_dissent"):
        """Profile dissent retrieval."""
        return profile_memory(operation, MemoryCategory.CONSENSUS_DISSENT)

    def add_result(self, result: MemoryProfileResult) -> None:
        """Add a profiling result."""
        self.profiles.append(result)

    def summary(self) -> dict[str, Any]:
        """Generate summary of all profiles."""
        if not self.profiles:
            return {"error": "No profiles recorded"}

        by_category: dict[str, list[MemoryProfileResult]] = {}
        for p in self.profiles:
            cat = p.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(p)

        summary = {}
        for cat, profiles in by_category.items():
            deltas = [p.delta_mb for p in profiles]
            peaks = [p.peak_delta_mb for p in profiles]

            summary[cat] = {
                "count": len(profiles),
                "avg_delta_mb": round(sum(deltas) / len(deltas), 2),
                "max_delta_mb": round(max(deltas), 2),
                "avg_peak_mb": round(sum(peaks) / len(peaks), 2),
                "max_peak_mb": round(max(peaks), 2),
            }

        return summary


# Global profiler instances for convenient access
km_profiler = KMMemoryProfiler()
consensus_profiler = ConsensusMemoryProfiler()


__all__ = [
    "MemoryProfiler",
    "MemoryProfileResult",
    "MemorySnapshot",
    "MemoryGrowthTracker",
    "MemoryCategory",
    "profile_memory",
    "track_memory",
    "KMMemoryProfiler",
    "ConsensusMemoryProfiler",
    "km_profiler",
    "consensus_profiler",
]
