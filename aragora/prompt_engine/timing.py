"""Timing models and helpers for prompt-engine latency profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

PROMPT_ENGINE_TARGET_DURATION_MS = 15_000.0
PROMPT_ENGINE_BOTTLENECK_BUDGET_SHARE = 0.15


@dataclass
class OperationTiming:
    """A measured prompt-engine operation."""

    operation: str
    duration_ms: float
    category: str = "compute"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the timing record."""
        data: dict[str, Any] = {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "category": self.category,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass
class PipelineTiming:
    """Latency summary for a prompt-engine pipeline run."""

    total_duration_ms: float = 0.0
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    operation_timings: list[OperationTiming] = field(default_factory=list)
    target_duration_ms: float = PROMPT_ENGINE_TARGET_DURATION_MS

    @property
    def is_within_target(self) -> bool:
        """Whether the pipeline stayed within its latency target."""
        return self.total_duration_ms <= self.target_duration_ms

    @property
    def overrun_ms(self) -> float:
        """How far beyond the target the pipeline ran."""
        return max(0.0, self.total_duration_ms - self.target_duration_ms)

    def top_operations(self, limit: int = 5) -> list[OperationTiming]:
        """Return the slowest measured operations."""
        return sorted(self.operation_timings, key=lambda item: item.duration_ms, reverse=True)[
            :limit
        ]

    def bottlenecks(
        self,
        limit: int = 5,
        min_budget_share: float = PROMPT_ENGINE_BOTTLENECK_BUDGET_SHARE,
    ) -> list[OperationTiming]:
        """Return operations consuming a meaningful share of the latency budget."""
        threshold_ms = max(1.0, self.target_duration_ms * min_budget_share)
        return [
            item
            for item in self.top_operations(limit=len(self.operation_timings))
            if item.duration_ms >= threshold_ms
        ][:limit]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the timing summary."""
        return {
            "total_duration_ms": round(self.total_duration_ms, 2),
            "target_duration_ms": round(self.target_duration_ms, 2),
            "is_within_target": self.is_within_target,
            "overrun_ms": round(self.overrun_ms, 2),
            "stage_durations_ms": {
                name: round(duration, 2) for name, duration in self.stage_durations_ms.items()
            },
            "operation_timings": [item.to_dict() for item in self.operation_timings],
            "bottlenecks": [item.to_dict() for item in self.bottlenecks()],
        }


def start_timer() -> float:
    """Start a monotonic timer."""
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    """Convert a timer start point into elapsed milliseconds."""
    return (time.perf_counter() - start) * 1000


def append_timing(
    timings: list[OperationTiming],
    operation: str,
    start: float,
    *,
    category: str = "compute",
    **metadata: Any,
) -> float:
    """Record an elapsed duration into a timing list."""
    duration_ms = elapsed_ms(start)
    timings.append(
        OperationTiming(
            operation=operation,
            duration_ms=duration_ms,
            category=category,
            metadata=metadata,
        )
    )
    return duration_ms


def format_timings(timings: list[OperationTiming]) -> str:
    """Render timing records for logs."""
    return ", ".join(f"{item.operation}={item.duration_ms:.1f}ms" for item in timings) or "none"
