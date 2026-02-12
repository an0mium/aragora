"""
Telemetry collection and aggregation for research integrations.

This module provides the TelemetryCollector class for buffering, aggregating,
and flushing telemetry events to a backend.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Protocol
import asyncio
import logging

from aragora.telemetry.research_events import TelemetryEvent, TelemetryEventType

logger = logging.getLogger(__name__)


class TelemetryBackend(Protocol):
    """Protocol for telemetry backends."""

    async def write(self, events: list[TelemetryEvent]) -> None:
        """Write events to backend storage."""
        ...


class InMemoryBackend:
    """Simple in-memory backend for testing."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    async def write(self, events: list[TelemetryEvent]) -> None:
        """Store events in memory."""
        self.events.extend(events)

    def get_events(
        self,
        event_type: TelemetryEventType | None = None,
    ) -> list[TelemetryEvent]:
        """Get stored events, optionally filtered by type."""
        if event_type is None:
            return self.events.copy()
        return [e for e in self.events if e.event_type == event_type]

    def clear(self) -> None:
        """Clear stored events."""
        self.events.clear()


class TelemetryCollector:
    """
    Collects and aggregates telemetry events.

    The collector buffers events and periodically flushes them to a backend.
    It also maintains running aggregates for key metrics.

    Example:
        backend = InMemoryBackend()
        collector = TelemetryCollector(backend=backend)

        # Record events
        await collector.record(stability_check_event(...))
        await collector.record(muse_calculation_event(...))

        # Get summary
        summary = collector.get_summary()
        print(f"Adaptive stopping stable rate: {summary['adaptive_stopping']['stable_rate']}")

        # Flush to backend
        await collector.flush()
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        flush_interval_seconds: int = 60,
        backend: TelemetryBackend | None = None,
    ):
        """Initialize the telemetry collector.

        Args:
            buffer_size: Maximum events to buffer before auto-flush
            flush_interval_seconds: Interval for periodic flush
            backend: Backend to write events to
        """
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds
        self.backend = backend
        self._buffer: list[TelemetryEvent] = []
        self._aggregates: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._flush_task: asyncio.Task[None] | None = None
        self._period_start = datetime.utcnow()

    async def start_periodic_flush(self) -> None:
        """Start background task for periodic flushing."""
        if self._flush_task is not None:
            return

        async def _periodic_flush() -> None:
            while True:
                await asyncio.sleep(self.flush_interval)
                try:
                    await self.flush()
                except Exception as e:
                    logger.error("periodic_flush_error: %s", e)

        self._flush_task = asyncio.create_task(_periodic_flush())

    async def stop_periodic_flush(self) -> None:
        """Stop background flush task."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

    async def record(self, event: TelemetryEvent) -> None:
        """Record a telemetry event.

        Args:
            event: The event to record
        """
        self._buffer.append(event)
        self._update_aggregates(event)

        if len(self._buffer) >= self.buffer_size:
            await self.flush()

    def _update_aggregates(self, event: TelemetryEvent) -> None:
        """Update running aggregates with new event."""
        key = event.event_type.value
        agg = self._aggregates[key]

        agg["count"] += 1

        if event.duration_ms is not None:
            agg["total_duration_ms"] += event.duration_ms
            agg["max_duration_ms"] = max(
                agg.get("max_duration_ms", 0),
                event.duration_ms,
            )
            agg["min_duration_ms"] = min(
                agg.get("min_duration_ms", float("inf")),
                event.duration_ms,
            )

        # Event-specific aggregates
        props = event.properties

        if event.event_type == TelemetryEventType.STABILITY_CHECK:
            if props.get("is_stable"):
                agg["stable_count"] += 1
            if props.get("muse_gated"):
                agg["muse_gated_count"] += 1
            if props.get("ascot_gated"):
                agg["ascot_gated_count"] += 1
            agg["stability_score_sum"] += props.get("stability_score", 0)

        elif event.event_type == TelemetryEventType.EARLY_TERMINATION:
            agg["rounds_saved_sum"] += props.get("rounds_saved", 0)

        elif event.event_type == TelemetryEventType.ROUTING_DECISION:
            mode = props.get("selected_mode", "unknown")
            agg[f"mode_{mode}_count"] += 1
            agg["doc_tokens_sum"] += props.get("doc_tokens", 0)

        elif event.event_type == TelemetryEventType.MUSE_CALCULATION:
            agg["divergence_sum"] += props.get("divergence_score", 0)
            agg["confidence_sum"] += props.get("consensus_confidence", 0)
            agg["subset_size_sum"] += props.get("subset_size", 0)

        elif event.event_type == TelemetryEventType.PRM_ERROR_DETECTED:
            if props.get("is_late_stage"):
                agg["late_stage_errors"] += 1

        elif event.event_type == TelemetryEventType.PRM_STEP_VERIFIED:
            verdict = props.get("verdict", "unknown")
            agg[f"verdict_{verdict}_count"] += 1

        elif event.event_type == TelemetryEventType.TEAM_COMPOSED:
            agg["diversity_score_sum"] += props.get("diversity_score", 0)
            agg["coverage_score_sum"] += props.get("coverage_score", 0)

        elif event.event_type == TelemetryEventType.CLAIM_VERIFIED:
            if props.get("verified"):
                agg["verified_count"] += 1

    async def flush(self) -> None:
        """Flush buffer to backend."""
        if not self._buffer:
            return

        events = self._buffer.copy()
        self._buffer.clear()

        if self.backend:
            try:
                await self.backend.write(events)
            except Exception as e:
                logger.error("telemetry_flush_error: %s", e)
                # Re-add events to buffer on failure (with limit)
                self._buffer = (events + self._buffer)[: self.buffer_size]

    def get_aggregates(self) -> dict[str, dict[str, float]]:
        """Get current aggregates.

        Returns:
            Mapping of event type to aggregate values
        """
        result: dict[str, dict[str, float]] = {}

        for event_type, agg in self._aggregates.items():
            result[event_type] = dict(agg)
            count = agg.get("count", 0)

            # Calculate averages
            if count > 0:
                if agg.get("total_duration_ms", 0) > 0:
                    result[event_type]["avg_duration_ms"] = agg["total_duration_ms"] / count

        return result

    def get_summary(self) -> dict[str, Any]:
        """Get human-readable summary of aggregates.

        Returns:
            Summary dict suitable for dashboards
        """
        aggs = self.get_aggregates()

        summary: dict[str, Any] = {
            "period_start": self._period_start.isoformat(),
            "period_end": datetime.utcnow().isoformat(),
            "period_seconds": (datetime.utcnow() - self._period_start).total_seconds(),
        }

        # Stability summary
        if "stability_check" in aggs:
            sc = aggs["stability_check"]
            count = sc.get("count", 0)
            summary["adaptive_stopping"] = {
                "checks": count,
                "stable_rate": sc.get("stable_count", 0) / count if count > 0 else 0,
                "muse_gate_rate": (sc.get("muse_gated_count", 0) / count if count > 0 else 0),
                "ascot_gate_rate": (sc.get("ascot_gated_count", 0) / count if count > 0 else 0),
                "avg_stability_score": (
                    sc.get("stability_score_sum", 0) / count if count > 0 else 0
                ),
            }

        # Early termination summary
        if "early_termination" in aggs:
            et = aggs["early_termination"]
            count = et.get("count", 0)
            summary["early_termination"] = {
                "total_terminations": count,
                "total_rounds_saved": et.get("rounds_saved_sum", 0),
                "avg_rounds_saved": (et.get("rounds_saved_sum", 0) / count if count > 0 else 0),
            }

        # Routing summary
        if "routing_decision" in aggs:
            rd = aggs["routing_decision"]
            total = rd.get("count", 0)
            if total > 0:
                summary["lara_routing"] = {
                    "total_queries": total,
                    "rag_pct": rd.get("mode_rag_count", 0) / total * 100,
                    "rlm_pct": rd.get("mode_rlm_count", 0) / total * 100,
                    "graph_pct": rd.get("mode_graph_count", 0) / total * 100,
                    "long_context_pct": (rd.get("mode_long_context_count", 0) / total * 100),
                    "hybrid_pct": rd.get("mode_hybrid_count", 0) / total * 100,
                    "avg_doc_tokens": rd.get("doc_tokens_sum", 0) / total,
                    "avg_latency_ms": rd.get("avg_duration_ms", 0),
                }

        # MUSE summary
        if "muse_calculation" in aggs:
            mc = aggs["muse_calculation"]
            count = mc.get("count", 0)
            if count > 0:
                summary["muse"] = {
                    "calculations": count,
                    "avg_divergence": mc.get("divergence_sum", 0) / count,
                    "avg_confidence": mc.get("confidence_sum", 0) / count,
                    "avg_subset_size": mc.get("subset_size_sum", 0) / count,
                    "avg_latency_ms": mc.get("avg_duration_ms", 0),
                }

        # PRM summary
        prm_verified = aggs.get("prm_step_verified", {})
        prm_errors = aggs.get("prm_error_detected", {})
        verified_count = prm_verified.get("count", 0)
        error_count = prm_errors.get("count", 0)

        if verified_count > 0 or error_count > 0:
            summary["think_prm"] = {
                "steps_verified": verified_count,
                "errors_detected": error_count,
                "late_stage_errors": prm_errors.get("late_stage_errors", 0),
                "error_rate": (error_count / verified_count if verified_count > 0 else 0),
                "late_stage_pct": (
                    prm_errors.get("late_stage_errors", 0) / error_count * 100
                    if error_count > 0
                    else 0
                ),
            }

        # Team composition summary
        if "team_composed" in aggs:
            tc = aggs["team_composed"]
            count = tc.get("count", 0)
            if count > 0:
                summary["a_hmad"] = {
                    "teams_composed": count,
                    "avg_diversity_score": tc.get("diversity_score_sum", 0) / count,
                    "avg_coverage_score": tc.get("coverage_score_sum", 0) / count,
                }

        # Claim verification summary
        if "claim_verified" in aggs:
            cv = aggs["claim_verified"]
            count = cv.get("count", 0)
            if count > 0:
                summary["claimcheck"] = {
                    "claims_verified": count,
                    "verified_rate": cv.get("verified_count", 0) / count,
                }

        return summary

    def reset_aggregates(self) -> None:
        """Reset all aggregates (typically after a period)."""
        self._aggregates.clear()
        self._period_start = datetime.utcnow()


# Singleton collector for easy access
_default_collector: TelemetryCollector | None = None


def get_telemetry_collector() -> TelemetryCollector:
    """Get the default telemetry collector singleton."""
    global _default_collector
    if _default_collector is None:
        _default_collector = TelemetryCollector()
    return _default_collector


def set_telemetry_collector(collector: TelemetryCollector) -> None:
    """Set the default telemetry collector."""
    global _default_collector
    _default_collector = collector


async def record_event(event: TelemetryEvent) -> None:
    """Convenience function to record an event to the default collector."""
    collector = get_telemetry_collector()
    await collector.record(event)
