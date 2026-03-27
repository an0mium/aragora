"""Timing models and helpers for prompt-engine latency profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

PROMPT_ENGINE_TARGET_DURATION_MS = 15_000.0
PROMPT_ENGINE_BOTTLENECK_BUDGET_SHARE = 0.15
PROMPT_ENGINE_PROFILE_TARGET_SHARE = 0.05

_CATEGORY_OPTIMIZATION_HINTS = {
    "llm": "Reduce prompt size, model latency, or round trips.",
    "io": "Cache or narrow knowledge/context lookups.",
    "compute": "Trim parsing and post-processing work.",
    "setup": "Reuse warm clients and avoid repeated initialization.",
}


def _share(value: float, total: float) -> float:
    """Return a bounded ratio for serialization."""
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, value / total))


def optimization_hint(category: str) -> str:
    """Return a terse optimization hint for a timing category."""
    return _CATEGORY_OPTIMIZATION_HINTS.get(category, "Inspect this operation for avoidable work.")


@dataclass
class OperationTiming:
    """A measured prompt-engine operation."""

    operation: str
    duration_ms: float
    category: str = "compute"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stage(self) -> str:
        """Best-effort stage derived from the operation name."""
        if "." not in self.operation:
            return "unknown"
        return self.operation.split(".", 1)[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the timing record."""
        data: dict[str, Any] = {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "category": self.category,
            "stage": self.stage,
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

    @property
    def tracked_duration_ms(self) -> float:
        """How much of the run is covered by explicit operation timings."""
        return sum(item.duration_ms for item in self.operation_timings)

    @property
    def untracked_duration_ms(self) -> float:
        """How much wall-clock time still lacks explicit operation coverage."""
        return max(0.0, self.total_duration_ms - self.tracked_duration_ms)

    @property
    def tracking_coverage(self) -> float:
        """Fraction of total run time covered by operation timings."""
        return _share(self.tracked_duration_ms, self.total_duration_ms)

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

    def stage_operations(self, stage: str) -> list[OperationTiming]:
        """Return all measured operations for a stage."""
        return [item for item in self.operation_timings if item.stage == stage]

    def stage_breakdown(self) -> list[dict[str, Any]]:
        """Return stages ordered by runtime with their hottest operation."""
        ordered = sorted(
            self.stage_durations_ms.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        breakdown: list[dict[str, Any]] = []
        for stage, duration_ms in ordered:
            operations = self.stage_operations(stage)
            top_operation = max(operations, key=lambda item: item.duration_ms, default=None)
            breakdown.append(
                {
                    "stage": stage,
                    "duration_ms": round(duration_ms, 2),
                    "share_of_total_pct": round(
                        _share(duration_ms, self.total_duration_ms) * 100, 2
                    ),
                    "operation_count": len(operations),
                    "top_operation": self._serialize_operation(
                        top_operation,
                        stage_duration_ms=duration_ms,
                    )
                    if top_operation
                    else None,
                }
            )
        return breakdown

    def optimization_targets(
        self,
        limit: int = 5,
        min_share_of_total: float = PROMPT_ENGINE_PROFILE_TARGET_SHARE,
    ) -> list[dict[str, Any]]:
        """Return the most actionable optimization targets for the run."""
        targets: list[dict[str, Any]] = []
        for item in self.top_operations(limit=len(self.operation_timings)):
            share_of_total = _share(item.duration_ms, self.total_duration_ms)
            if share_of_total < min_share_of_total and targets:
                continue
            targets.append(self._serialize_operation(item))
            if len(targets) >= limit:
                break
        return targets

    def _serialize_operation(
        self,
        item: OperationTiming,
        *,
        stage_duration_ms: float | None = None,
    ) -> dict[str, Any]:
        """Serialize an operation with contextual percentage data."""
        data = item.to_dict()
        stage_total = stage_duration_ms
        if stage_total is None:
            stage_total = self.stage_durations_ms.get(item.stage, 0.0)
        data["share_of_total_pct"] = round(
            _share(item.duration_ms, self.total_duration_ms) * 100, 2
        )
        data["share_of_stage_pct"] = round(_share(item.duration_ms, stage_total) * 100, 2)
        data["optimization_hint"] = optimization_hint(item.category)
        return data

    def to_dict(self) -> dict[str, Any]:
        """Serialize the timing summary."""
        return {
            "total_duration_ms": round(self.total_duration_ms, 2),
            "target_duration_ms": round(self.target_duration_ms, 2),
            "is_within_target": self.is_within_target,
            "overrun_ms": round(self.overrun_ms, 2),
            "tracked_duration_ms": round(self.tracked_duration_ms, 2),
            "untracked_duration_ms": round(self.untracked_duration_ms, 2),
            "tracking_coverage_pct": round(self.tracking_coverage * 100, 2),
            "stage_durations_ms": {
                name: round(duration, 2) for name, duration in self.stage_durations_ms.items()
            },
            "stage_breakdown": self.stage_breakdown(),
            "operation_timings": [
                self._serialize_operation(item) for item in self.operation_timings
            ],
            "bottlenecks": [self._serialize_operation(item) for item in self.bottlenecks()],
            "optimization_targets": self.optimization_targets(),
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
