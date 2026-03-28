"""
Per-request AI model usage telemetry pipeline.

Captures tokens consumed, latency measurements, model version information,
and cost calculations during debate sessions. Integrates with the existing
TelemetryCollector for buffered event emission and aggregation.

Usage:
    pipeline = ModelUsagePipeline()

    # Record individual model calls
    pipeline.record_model_call(
        debate_id="debate-123",
        agent_id="agent-anthropic",
        provider="anthropic",
        model="claude-sonnet-4.6",
        tokens_in=1500,
        tokens_out=800,
        latency_ms=1240.5,
        operation="proposal",
    )

    # Get debate-level summary
    summary = pipeline.get_debate_summary("debate-123")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from aragora.billing.usage import calculate_token_cost
from aragora.telemetry.research_events import TelemetryEvent, TelemetryEventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended event types for model usage telemetry
# ---------------------------------------------------------------------------


class ModelUsageEventType(str, Enum):
    """Event types specific to model usage telemetry."""

    MODEL_CALL = "model_call"
    DEBATE_COST_SUMMARY = "debate_cost_summary"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelCallRecord:
    """A single model API call record with full attribution."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Attribution
    debate_id: str = ""
    agent_id: str = ""
    workspace_id: str = ""
    round_number: int | None = None

    # Model info
    provider: str = ""
    model: str = ""
    model_version: str = ""

    # Token counts
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0

    # Timing
    latency_ms: float = 0.0

    # Cost (computed)
    cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))

    # Context
    operation: str = ""  # "proposal", "critique", "revision", "consensus", "analysis"
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_cost(self) -> Decimal:
        """Calculate cost from provider pricing tables."""
        self.cost_usd = calculate_token_cost(
            self.provider, self.model, self.tokens_in, self.tokens_out
        )
        return self.cost_usd

    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "debate_id": self.debate_id,
            "agent_id": self.agent_id,
            "workspace_id": self.workspace_id,
            "round_number": self.round_number,
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "latency_ms": self.latency_ms,
            "cost_usd": str(self.cost_usd),
            "operation": self.operation,
            "metadata": self.metadata,
        }


@dataclass
class DebateUsageSummary:
    """Aggregated model usage for a single debate session."""

    debate_id: str = ""
    workspace_id: str = ""

    # Totals
    total_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tokens_cached: int = 0
    total_cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))

    # Latency stats
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    # Breakdowns
    by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_operation: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Timeline
    first_call: datetime | None = None
    last_call: datetime | None = None

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_tokens_in + self.total_tokens_out

    @property
    def wall_time_ms(self) -> float:
        """Wall-clock time from first to last call."""
        if self.first_call is None or self.last_call is None:
            return 0.0
        delta = self.last_call - self.first_call
        return delta.total_seconds() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "debate_id": self.debate_id,
            "workspace_id": self.workspace_id,
            "total_calls": self.total_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens_cached": self.total_tokens_cached,
            "total_tokens": self.total_tokens,
            "total_cost_usd": str(self.total_cost_usd),
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.total_calls > 0 else 0.0,
            "max_latency_ms": self.max_latency_ms,
            "wall_time_ms": self.wall_time_ms,
            "by_agent": self.by_agent,
            "by_model": self.by_model,
            "by_operation": self.by_operation,
            "first_call": self.first_call.isoformat() if self.first_call else None,
            "last_call": self.last_call.isoformat() if self.last_call else None,
        }


# ---------------------------------------------------------------------------
# Telemetry event constructors
# ---------------------------------------------------------------------------


def model_call_event(record: ModelCallRecord) -> TelemetryEvent:
    """Create a TelemetryEvent from a model call record.

    This bridges ModelCallRecord into the existing TelemetryCollector pipeline
    by using the BENCHMARK_RUN event type as the closest generic fit, with
    a ``model_usage_type`` property to distinguish it.
    """
    return TelemetryEvent(
        event_type=TelemetryEventType.BENCHMARK_RUN,
        timestamp=record.timestamp,
        debate_id=record.debate_id or None,
        workspace_id=record.workspace_id or None,
        round_number=record.round_number,
        properties={
            "model_usage_type": ModelUsageEventType.MODEL_CALL.value,
            "record_id": record.id,
            "agent_id": record.agent_id,
            "provider": record.provider,
            "model": record.model,
            "model_version": record.model_version,
            "tokens_in": record.tokens_in,
            "tokens_out": record.tokens_out,
            "tokens_cached": record.tokens_cached,
            "cost_usd": str(record.cost_usd),
            "operation": record.operation,
        },
        duration_ms=record.latency_ms,
    )


def debate_cost_summary_event(summary: DebateUsageSummary) -> TelemetryEvent:
    """Create a TelemetryEvent from a debate usage summary."""
    return TelemetryEvent(
        event_type=TelemetryEventType.BENCHMARK_RUN,
        debate_id=summary.debate_id or None,
        workspace_id=summary.workspace_id or None,
        properties={
            "model_usage_type": ModelUsageEventType.DEBATE_COST_SUMMARY.value,
            "total_calls": summary.total_calls,
            "total_tokens_in": summary.total_tokens_in,
            "total_tokens_out": summary.total_tokens_out,
            "total_tokens": summary.total_tokens,
            "total_cost_usd": str(summary.total_cost_usd),
            "avg_latency_ms": summary.avg_latency_ms,
            "models_used": list(summary.by_model.keys()),
            "agents_used": list(summary.by_agent.keys()),
        },
        duration_ms=summary.total_latency_ms,
    )


# ---------------------------------------------------------------------------
# Model Usage Pipeline
# ---------------------------------------------------------------------------


def _empty_breakdown() -> dict[str, Any]:
    return {
        "calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": Decimal("0"),
        "latency_ms": 0.0,
    }


class ModelUsagePipeline:
    """Aggregates per-request model usage into debate-level summaries.

    Thread-safe for single-writer usage (typical debate orchestration).
    All calls to ``record_model_call`` accumulate into per-debate summaries
    which can be retrieved via ``get_debate_summary``.
    """

    def __init__(self) -> None:
        self._records: dict[str, list[ModelCallRecord]] = defaultdict(list)
        self._summaries: dict[str, DebateUsageSummary] = {}

    def record_model_call(
        self,
        debate_id: str,
        agent_id: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        operation: str = "",
        *,
        model_version: str = "",
        tokens_cached: int = 0,
        workspace_id: str = "",
        round_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelCallRecord:
        """Record a single model API call and update the debate summary.

        Args:
            debate_id: ID of the debate session.
            agent_id: ID of the agent making the call.
            provider: Model provider (e.g. "anthropic", "openai").
            model: Model name (e.g. "claude-sonnet-4.6").
            tokens_in: Input tokens consumed.
            tokens_out: Output tokens generated.
            latency_ms: Request latency in milliseconds.
            operation: Call purpose (e.g. "proposal", "critique").
            model_version: Specific model version string.
            tokens_cached: Cached/prompt-caching tokens.
            workspace_id: Workspace for attribution.
            round_number: Debate round number if applicable.
            metadata: Additional key-value metadata.

        Returns:
            The created ModelCallRecord.
        """
        record = ModelCallRecord(
            debate_id=debate_id,
            agent_id=agent_id,
            provider=provider,
            model=model,
            model_version=model_version,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_cached=tokens_cached,
            latency_ms=latency_ms,
            operation=operation,
            workspace_id=workspace_id,
            round_number=round_number,
            metadata=metadata or {},
        )
        record.calculate_cost()

        self._records[debate_id].append(record)
        self._update_summary(record)

        return record

    def _update_summary(self, record: ModelCallRecord) -> None:
        """Incrementally update the debate summary with a new record."""
        debate_id = record.debate_id
        if debate_id not in self._summaries:
            self._summaries[debate_id] = DebateUsageSummary(
                debate_id=debate_id,
                workspace_id=record.workspace_id,
            )

        s = self._summaries[debate_id]
        s.total_calls += 1
        s.total_tokens_in += record.tokens_in
        s.total_tokens_out += record.tokens_out
        s.total_tokens_cached += record.tokens_cached
        s.total_cost_usd += record.cost_usd
        s.total_latency_ms += record.latency_ms
        s.min_latency_ms = min(s.min_latency_ms, record.latency_ms)
        s.max_latency_ms = max(s.max_latency_ms, record.latency_ms)

        # Timeline
        if s.first_call is None or record.timestamp < s.first_call:
            s.first_call = record.timestamp
        if s.last_call is None or record.timestamp > s.last_call:
            s.last_call = record.timestamp

        # By-agent breakdown
        agent_key = record.agent_id
        if agent_key not in s.by_agent:
            s.by_agent[agent_key] = _empty_breakdown()
        ab = s.by_agent[agent_key]
        ab["calls"] += 1
        ab["tokens_in"] += record.tokens_in
        ab["tokens_out"] += record.tokens_out
        ab["cost_usd"] += record.cost_usd
        ab["latency_ms"] += record.latency_ms

        # By-model breakdown
        model_key = f"{record.provider}/{record.model}"
        if model_key not in s.by_model:
            s.by_model[model_key] = _empty_breakdown()
        mb = s.by_model[model_key]
        mb["calls"] += 1
        mb["tokens_in"] += record.tokens_in
        mb["tokens_out"] += record.tokens_out
        mb["cost_usd"] += record.cost_usd
        mb["latency_ms"] += record.latency_ms

        # By-operation breakdown
        if record.operation:
            if record.operation not in s.by_operation:
                s.by_operation[record.operation] = _empty_breakdown()
            ob = s.by_operation[record.operation]
            ob["calls"] += 1
            ob["tokens_in"] += record.tokens_in
            ob["tokens_out"] += record.tokens_out
            ob["cost_usd"] += record.cost_usd
            ob["latency_ms"] += record.latency_ms

    def get_debate_summary(self, debate_id: str) -> DebateUsageSummary | None:
        """Get the aggregated usage summary for a debate.

        Returns None if no records exist for the given debate.
        """
        return self._summaries.get(debate_id)

    def get_debate_records(self, debate_id: str) -> list[ModelCallRecord]:
        """Get all individual call records for a debate."""
        return list(self._records.get(debate_id, []))

    def get_all_debate_ids(self) -> list[str]:
        """List all debate IDs with recorded usage."""
        return list(self._summaries.keys())

    def clear_debate(self, debate_id: str) -> None:
        """Remove all records and summary for a debate."""
        self._records.pop(debate_id, None)
        self._summaries.pop(debate_id, None)

    def clear_all(self) -> None:
        """Remove all records and summaries."""
        self._records.clear()
        self._summaries.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_pipeline: ModelUsagePipeline | None = None


def get_model_usage_pipeline() -> ModelUsagePipeline:
    """Get or create the default ModelUsagePipeline singleton."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ModelUsagePipeline()
    return _default_pipeline


def set_model_usage_pipeline(pipeline: ModelUsagePipeline) -> None:
    """Replace the default pipeline (useful in tests)."""
    global _default_pipeline
    _default_pipeline = pipeline
