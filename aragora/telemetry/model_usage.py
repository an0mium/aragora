"""
Model usage telemetry pipeline for AI model request tracking.

Captures per-request metrics (tokens, latency, model version, costs) during
debate sessions and provides storage, querying, and aggregation.

Usage:
    pipeline = ModelUsagePipeline()

    # Record a model request
    record = pipeline.record(
        agent_name="claude",
        provider="anthropic",
        model="claude-opus-4.6",
        tokens_in=1500,
        tokens_out=800,
        latency_ms=1234.5,
        debate_id="debate-123",
        round_number=2,
        operation="proposal",
    )

    # Aggregate by model
    agg = pipeline.aggregate_by_model()
    # {'claude-opus-4.6': ModelUsageAggregate(total_requests=1, ...)}

    # Query records
    records = pipeline.query(debate_id="debate-123")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from aragora.billing.usage import calculate_token_cost

logger = logging.getLogger(__name__)


@dataclass
class ModelUsageRecord:
    """A single AI model request's telemetry data."""

    id: str = field(default_factory=lambda: str(uuid4()))
    agent_name: str = ""
    provider: str = ""
    model: str = ""

    # Token metrics
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0

    # Performance
    latency_ms: float = 0.0

    # Cost (calculated from tokens + pricing)
    cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))

    # Context
    debate_id: str = ""
    round_number: int | None = None
    operation: str = ""  # proposal, critique, revision, vote, etc.

    # Timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_cost(self) -> Decimal:
        """Calculate and set cost based on tokens and provider pricing."""
        self.cost_usd = calculate_token_cost(
            self.provider,
            self.model,
            self.tokens_in,
            self.tokens_out,
            self.tokens_cached,
        )
        return self.cost_usd

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output + cached)."""
        return self.tokens_in + self.tokens_out + self.tokens_cached

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "provider": self.provider,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "latency_ms": self.latency_ms,
            "cost_usd": str(self.cost_usd),
            "debate_id": self.debate_id,
            "round_number": self.round_number,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelUsageRecord:
        """Deserialize from dictionary."""
        record = cls(
            id=data.get("id", str(uuid4())),
            agent_name=data.get("agent_name", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            tokens_cached=data.get("tokens_cached", 0),
            latency_ms=data.get("latency_ms", 0.0),
            cost_usd=Decimal(data.get("cost_usd", "0")),
            debate_id=data.get("debate_id", ""),
            round_number=data.get("round_number"),
            operation=data.get("operation", ""),
            metadata=data.get("metadata", {}),
        )
        if "timestamp" in data and data["timestamp"]:
            if isinstance(data["timestamp"], str):
                record.timestamp = datetime.fromisoformat(data["timestamp"])
            else:
                record.timestamp = data["timestamp"]
        return record


@dataclass
class ModelUsageAggregate:
    """Aggregated model usage statistics."""

    total_requests: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tokens_cached: int = 0
    total_cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))

    # Latency stats
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    # Models and providers seen
    models_seen: set[str] = field(default_factory=set)
    providers_seen: set[str] = field(default_factory=set)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across requests."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def avg_tokens_per_request(self) -> float:
        """Average total tokens per request."""
        if self.total_requests == 0:
            return 0.0
        total = self.total_tokens_in + self.total_tokens_out + self.total_tokens_cached
        return total / self.total_requests

    @property
    def avg_cost_per_request(self) -> Decimal:
        """Average cost per request."""
        if self.total_requests == 0:
            return Decimal("0")
        return self.total_cost_usd / self.total_requests

    def add_record(self, record: ModelUsageRecord) -> None:
        """Incorporate a record into the aggregate."""
        self.total_requests += 1
        self.total_tokens_in += record.tokens_in
        self.total_tokens_out += record.tokens_out
        self.total_tokens_cached += record.tokens_cached
        self.total_cost_usd += record.cost_usd
        self.total_latency_ms += record.latency_ms
        if record.latency_ms > 0:
            self.min_latency_ms = min(self.min_latency_ms, record.latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, record.latency_ms)
        if record.model:
            self.models_seen.add(record.model)
        if record.provider:
            self.providers_seen.add(record.provider)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens_cached": self.total_tokens_cached,
            "total_cost_usd": str(self.total_cost_usd),
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.total_requests > 0 else 0.0,
            "max_latency_ms": self.max_latency_ms,
            "avg_tokens_per_request": self.avg_tokens_per_request,
            "avg_cost_per_request": str(self.avg_cost_per_request),
            "models_seen": sorted(self.models_seen),
            "providers_seen": sorted(self.providers_seen),
        }


class ModelUsageStore:
    """Thread-safe in-memory store for model usage records.

    Records are stored in memory with configurable maximum capacity.
    Older records are evicted when the capacity is exceeded.
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._records: list[ModelUsageRecord] = []
        self._lock = threading.Lock()
        self.max_records = max_records

    def add(self, record: ModelUsageRecord) -> None:
        """Add a record to the store."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_records:
                # Evict oldest records
                overflow = len(self._records) - self.max_records
                self._records = self._records[overflow:]

    def query(
        self,
        *,
        debate_id: str | None = None,
        agent_name: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[ModelUsageRecord]:
        """Query records with optional filters.

        All filters are AND-combined. Returns records in chronological order.
        """
        with self._lock:
            results = self._records.copy()

        if debate_id is not None:
            results = [r for r in results if r.debate_id == debate_id]
        if agent_name is not None:
            results = [r for r in results if r.agent_name == agent_name]
        if model is not None:
            results = [r for r in results if r.model == model]
        if provider is not None:
            results = [r for r in results if r.provider == provider]
        if operation is not None:
            results = [r for r in results if r.operation == operation]
        if since is not None:
            results = [r for r in results if r.timestamp >= since]
        if until is not None:
            results = [r for r in results if r.timestamp <= until]
        if limit is not None:
            results = results[:limit]

        return results

    def count(self) -> int:
        """Number of records in the store."""
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        """Remove all records."""
        with self._lock:
            self._records.clear()

    def get_all(self) -> list[ModelUsageRecord]:
        """Return a copy of all records."""
        with self._lock:
            return self._records.copy()


class ModelUsageAggregator:
    """Aggregates model usage records by various dimensions."""

    @staticmethod
    def _aggregate(records: list[ModelUsageRecord]) -> ModelUsageAggregate:
        """Build an aggregate from a list of records."""
        agg = ModelUsageAggregate()
        for record in records:
            agg.add_record(record)
        return agg

    @staticmethod
    def by_model(records: list[ModelUsageRecord]) -> dict[str, ModelUsageAggregate]:
        """Aggregate records grouped by model name."""
        groups: dict[str, list[ModelUsageRecord]] = defaultdict(list)
        for r in records:
            groups[r.model].append(r)
        return {model: ModelUsageAggregator._aggregate(recs) for model, recs in groups.items()}

    @staticmethod
    def by_agent(records: list[ModelUsageRecord]) -> dict[str, ModelUsageAggregate]:
        """Aggregate records grouped by agent name."""
        groups: dict[str, list[ModelUsageRecord]] = defaultdict(list)
        for r in records:
            groups[r.agent_name].append(r)
        return {agent: ModelUsageAggregator._aggregate(recs) for agent, recs in groups.items()}

    @staticmethod
    def by_provider(records: list[ModelUsageRecord]) -> dict[str, ModelUsageAggregate]:
        """Aggregate records grouped by provider."""
        groups: dict[str, list[ModelUsageRecord]] = defaultdict(list)
        for r in records:
            groups[r.provider].append(r)
        return {
            provider: ModelUsageAggregator._aggregate(recs) for provider, recs in groups.items()
        }

    @staticmethod
    def by_debate(records: list[ModelUsageRecord]) -> dict[str, ModelUsageAggregate]:
        """Aggregate records grouped by debate ID."""
        groups: dict[str, list[ModelUsageRecord]] = defaultdict(list)
        for r in records:
            if r.debate_id:
                groups[r.debate_id].append(r)
        return {
            debate_id: ModelUsageAggregator._aggregate(recs) for debate_id, recs in groups.items()
        }

    @staticmethod
    def by_operation(records: list[ModelUsageRecord]) -> dict[str, ModelUsageAggregate]:
        """Aggregate records grouped by operation type."""
        groups: dict[str, list[ModelUsageRecord]] = defaultdict(list)
        for r in records:
            groups[r.operation].append(r)
        return {op: ModelUsageAggregator._aggregate(recs) for op, recs in groups.items()}

    @staticmethod
    def total(records: list[ModelUsageRecord]) -> ModelUsageAggregate:
        """Compute a single aggregate across all records."""
        return ModelUsageAggregator._aggregate(records)


class ModelUsagePipeline:
    """Unified pipeline for collecting, storing, and aggregating model usage.

    Provides the main entry point for recording AI model requests and
    retrieving usage analytics.

    Example:
        pipeline = ModelUsagePipeline()
        pipeline.record(
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4.6",
            tokens_in=1500,
            tokens_out=800,
            latency_ms=1234.5,
            debate_id="debate-123",
        )
        summary = pipeline.get_summary()
    """

    def __init__(
        self,
        store: ModelUsageStore | None = None,
        max_records: int = 10_000,
    ) -> None:
        self._store = store or ModelUsageStore(max_records=max_records)
        self._aggregator = ModelUsageAggregator()

    @property
    def store(self) -> ModelUsageStore:
        """Access the underlying store."""
        return self._store

    def record(
        self,
        *,
        agent_name: str = "",
        provider: str = "",
        model: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_cached: int = 0,
        latency_ms: float = 0.0,
        debate_id: str = "",
        round_number: int | None = None,
        operation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ModelUsageRecord:
        """Record a single AI model request.

        Automatically calculates cost from the provider pricing table.

        Returns:
            The created ModelUsageRecord.
        """
        record = ModelUsageRecord(
            agent_name=agent_name,
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_cached=tokens_cached,
            latency_ms=latency_ms,
            debate_id=debate_id,
            round_number=round_number,
            operation=operation,
            metadata=metadata or {},
        )
        record.calculate_cost()
        self._store.add(record)

        logger.debug(
            "model_usage_recorded agent=%s model=%s tokens=%d latency=%.1fms cost=$%s",
            agent_name,
            model,
            record.total_tokens,
            latency_ms,
            record.cost_usd,
        )

        return record

    def record_from_dict(self, data: dict[str, Any]) -> ModelUsageRecord:
        """Record from a pre-built dictionary."""
        record = ModelUsageRecord.from_dict(data)
        if record.cost_usd == Decimal("0") and record.total_tokens > 0:
            record.calculate_cost()
        self._store.add(record)
        return record

    def query(self, **kwargs: Any) -> list[ModelUsageRecord]:
        """Query stored records. See ModelUsageStore.query for parameters."""
        return self._store.query(**kwargs)

    def aggregate_by_model(self, **query_kwargs: Any) -> dict[str, ModelUsageAggregate]:
        """Aggregate filtered records by model."""
        records = self._store.query(**query_kwargs) if query_kwargs else self._store.get_all()
        return self._aggregator.by_model(records)

    def aggregate_by_agent(self, **query_kwargs: Any) -> dict[str, ModelUsageAggregate]:
        """Aggregate filtered records by agent."""
        records = self._store.query(**query_kwargs) if query_kwargs else self._store.get_all()
        return self._aggregator.by_agent(records)

    def aggregate_by_provider(self, **query_kwargs: Any) -> dict[str, ModelUsageAggregate]:
        """Aggregate filtered records by provider."""
        records = self._store.query(**query_kwargs) if query_kwargs else self._store.get_all()
        return self._aggregator.by_provider(records)

    def aggregate_by_debate(self, **query_kwargs: Any) -> dict[str, ModelUsageAggregate]:
        """Aggregate filtered records by debate."""
        records = self._store.query(**query_kwargs) if query_kwargs else self._store.get_all()
        return self._aggregator.by_debate(records)

    def aggregate_by_operation(self, **query_kwargs: Any) -> dict[str, ModelUsageAggregate]:
        """Aggregate filtered records by operation type."""
        records = self._store.query(**query_kwargs) if query_kwargs else self._store.get_all()
        return self._aggregator.by_operation(records)

    def get_total(self, **query_kwargs: Any) -> ModelUsageAggregate:
        """Get a single aggregate across all (optionally filtered) records."""
        records = self._store.query(**query_kwargs) if query_kwargs else self._store.get_all()
        return self._aggregator.total(records)

    def get_summary(self, **query_kwargs: Any) -> dict[str, Any]:
        """Get a human-readable summary of model usage.

        Returns a dict suitable for dashboards or JSON serialization.
        """
        records = self._store.query(**query_kwargs) if query_kwargs else self._store.get_all()
        total = self._aggregator.total(records)
        by_model = self._aggregator.by_model(records)
        by_agent = self._aggregator.by_agent(records)
        by_provider = self._aggregator.by_provider(records)

        return {
            "total": total.to_dict(),
            "by_model": {k: v.to_dict() for k, v in by_model.items()},
            "by_agent": {k: v.to_dict() for k, v in by_agent.items()},
            "by_provider": {k: v.to_dict() for k, v in by_provider.items()},
            "record_count": len(records),
        }

    def clear(self) -> None:
        """Clear all stored records."""
        self._store.clear()


class ModelUsageTimer:
    """Context manager for timing model requests and recording usage.

    Example:
        pipeline = ModelUsagePipeline()
        with ModelUsageTimer(pipeline, agent_name="claude", provider="anthropic",
                             model="claude-opus-4.6") as timer:
            result = call_model(prompt)
            timer.set_tokens(tokens_in=1500, tokens_out=800)
    """

    def __init__(
        self,
        pipeline: ModelUsagePipeline,
        *,
        agent_name: str = "",
        provider: str = "",
        model: str = "",
        debate_id: str = "",
        round_number: int | None = None,
        operation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._agent_name = agent_name
        self._provider = provider
        self._model = model
        self._debate_id = debate_id
        self._round_number = round_number
        self._operation = operation
        self._metadata = metadata or {}
        self._tokens_in = 0
        self._tokens_out = 0
        self._tokens_cached = 0
        self._start_time = 0.0
        self._record: ModelUsageRecord | None = None

    def set_tokens(
        self,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_cached: int = 0,
    ) -> None:
        """Set token counts (call before exiting the context)."""
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._tokens_cached = tokens_cached

    @property
    def record(self) -> ModelUsageRecord | None:
        """The recorded ModelUsageRecord, available after context exit."""
        return self._record

    def __enter__(self) -> ModelUsageTimer:
        self._start_time = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        self._record = self._pipeline.record(
            agent_name=self._agent_name,
            provider=self._provider,
            model=self._model,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            tokens_cached=self._tokens_cached,
            latency_ms=elapsed_ms,
            debate_id=self._debate_id,
            round_number=self._round_number,
            operation=self._operation,
            metadata=self._metadata,
        )


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_default_pipeline: ModelUsagePipeline | None = None
_pipeline_lock = threading.Lock()


def get_model_usage_pipeline() -> ModelUsagePipeline:
    """Get or create the default global pipeline singleton."""
    global _default_pipeline
    if _default_pipeline is None:
        with _pipeline_lock:
            if _default_pipeline is None:
                _default_pipeline = ModelUsagePipeline()
    return _default_pipeline


def set_model_usage_pipeline(pipeline: ModelUsagePipeline) -> None:
    """Replace the global pipeline singleton."""
    global _default_pipeline
    with _pipeline_lock:
        _default_pipeline = pipeline


def reset_model_usage_pipeline() -> None:
    """Reset the global pipeline singleton (useful for tests)."""
    global _default_pipeline
    with _pipeline_lock:
        _default_pipeline = None


__all__ = [
    "ModelUsageRecord",
    "ModelUsageAggregate",
    "ModelUsageStore",
    "ModelUsageAggregator",
    "ModelUsagePipeline",
    "ModelUsageTimer",
    "get_model_usage_pipeline",
    "set_model_usage_pipeline",
    "reset_model_usage_pipeline",
]
