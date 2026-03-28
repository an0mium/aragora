"""
Backend telemetry pipeline for AI model usage tracking.

Captures per-request AI model usage metrics during debate sessions:
- Token counts (input, output, cached)
- Latency (ms)
- Model version / provider
- Cost (USD)

Provides data collection, storage, and aggregation capabilities.
"""

from __future__ import annotations

__all__ = [
    "ModelUsageRecord",
    "ModelUsageStore",
    "ModelUsageCollector",
    "ModelUsageAggregator",
    "AggregationWindow",
    "track_model_call",
]

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelUsageRecord:
    """A single AI model invocation record."""

    record_id: str
    timestamp: str  # ISO-8601
    debate_id: str
    session_id: str
    agent_id: str
    agent_name: str
    provider: str  # e.g. "anthropic", "openai", "mistral"
    model: str  # e.g. "claude-opus-4-6", "gpt-4.1"
    model_version: str  # provider-reported version or ""
    operation: str  # "propose", "critique", "vote", "revise"
    tokens_in: int
    tokens_out: int
    tokens_cached: int
    latency_ms: float
    cost_usd: str  # stored as string for lossless Decimal round-trip
    success: bool
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class AggregationWindow(str, Enum):
    """Time windows for metric aggregation."""

    MINUTE = "minute"
    HOUR = "hour"
    DEBATE = "debate"
    SESSION = "session"


# ---------------------------------------------------------------------------
# Storage protocol + in-memory implementation
# ---------------------------------------------------------------------------


class ModelUsageBackend(Protocol):
    """Protocol for pluggable model usage persistence."""

    async def store(self, records: list[ModelUsageRecord]) -> None: ...

    async def query(
        self,
        *,
        debate_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[ModelUsageRecord]: ...


class ModelUsageStore:
    """In-memory store for model usage records with query support."""

    def __init__(self) -> None:
        self._records: list[ModelUsageRecord] = []
        self._lock = asyncio.Lock()

    async def store(self, records: list[ModelUsageRecord]) -> None:
        async with self._lock:
            self._records.extend(records)

    async def query(
        self,
        *,
        debate_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[ModelUsageRecord]:
        async with self._lock:
            results = self._records
            if debate_id is not None:
                results = [r for r in results if r.debate_id == debate_id]
            if session_id is not None:
                results = [r for r in results if r.session_id == session_id]
            if agent_id is not None:
                results = [r for r in results if r.agent_id == agent_id]
            if provider is not None:
                results = [r for r in results if r.provider == provider]
            if model is not None:
                results = [r for r in results if r.model == model]
            if since is not None:
                since_iso = since.isoformat()
                results = [r for r in results if r.timestamp >= since_iso]
            if until is not None:
                until_iso = until.isoformat()
                results = [r for r in results if r.timestamp <= until_iso]
            return results[:limit]

    async def count(self) -> int:
        async with self._lock:
            return len(self._records)

    async def clear(self) -> None:
        async with self._lock:
            self._records.clear()


# ---------------------------------------------------------------------------
# Collector — captures metrics and buffers for flush
# ---------------------------------------------------------------------------


class ModelUsageCollector:
    """Collects model usage records with buffering and periodic flush.

    Usage::

        store = ModelUsageStore()
        collector = ModelUsageCollector(backend=store)

        # Track a call via context manager
        async with collector.track_call(
            debate_id="d-1", session_id="s-1",
            agent_id="a-1", agent_name="claude",
            provider="anthropic", model="claude-opus-4-6",
            operation="propose",
        ) as ctx:
            # ... perform the LLM call ...
            ctx.set_tokens(tokens_in=500, tokens_out=200)
            ctx.set_cost("0.0125")

        await collector.flush()
    """

    def __init__(
        self,
        backend: ModelUsageBackend | ModelUsageStore | None = None,
        buffer_size: int = 500,
        flush_interval_seconds: int = 30,
    ) -> None:
        self._backend = backend
        self._buffer: list[ModelUsageRecord] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._flush_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start periodic background flush."""
        if self._flush_task is not None:
            return

        async def _loop() -> None:
            while True:
                await asyncio.sleep(self._flush_interval)
                try:
                    await self.flush()
                except Exception:
                    logger.exception("model_usage_flush_error")

        self._flush_task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        """Stop periodic flush and drain remaining buffer."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush()

    # -- recording -----------------------------------------------------------

    async def record(self, rec: ModelUsageRecord) -> None:
        """Add a record to the buffer; auto-flush when full."""
        async with self._lock:
            self._buffer.append(rec)
            if len(self._buffer) >= self._buffer_size:
                await self._flush_locked()

    @asynccontextmanager
    async def track_call(
        self,
        *,
        debate_id: str,
        session_id: str,
        agent_id: str,
        agent_name: str,
        provider: str,
        model: str,
        operation: str,
        model_version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[_TrackingContext]:
        """Context manager that times a model call and records the result."""
        ctx = _TrackingContext(
            debate_id=debate_id,
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            provider=provider,
            model=model,
            model_version=model_version,
            operation=operation,
            metadata=metadata or {},
        )
        try:
            yield ctx
        except Exception as exc:
            ctx.set_error(exc)
            raise
        finally:
            record = ctx.build_record()
            await self.record(record)

    # -- flush ---------------------------------------------------------------

    async def flush(self) -> int:
        """Flush buffered records to the backend. Returns count flushed."""
        async with self._lock:
            return await self._flush_locked()

    async def _flush_locked(self) -> int:
        if not self._buffer:
            return 0
        batch = self._buffer.copy()
        self._buffer.clear()
        if self._backend is not None:
            try:
                await self._backend.store(batch)
            except Exception:
                logger.exception("model_usage_backend_write_error")
                # Re-queue up to buffer_size
                self._buffer = (batch + self._buffer)[: self._buffer_size]
                return 0
        return len(batch)

    @property
    def pending(self) -> int:
        return len(self._buffer)


class _TrackingContext:
    """Mutable context handed to callers inside ``track_call``."""

    def __init__(
        self,
        *,
        debate_id: str,
        session_id: str,
        agent_id: str,
        agent_name: str,
        provider: str,
        model: str,
        model_version: str,
        operation: str,
        metadata: dict[str, Any],
    ) -> None:
        self._debate_id = debate_id
        self._session_id = session_id
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._provider = provider
        self._model = model
        self._model_version = model_version
        self._operation = operation
        self._metadata = metadata
        self._tokens_in = 0
        self._tokens_out = 0
        self._tokens_cached = 0
        self._cost_usd = "0"
        self._success = True
        self._error_type: str | None = None
        self._start = time.monotonic()

    def set_tokens(
        self,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_cached: int = 0,
    ) -> None:
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._tokens_cached = tokens_cached

    def set_cost(self, cost_usd: str | Decimal) -> None:
        self._cost_usd = str(cost_usd)

    def set_model_version(self, version: str) -> None:
        self._model_version = version

    def set_error(self, exc: Exception) -> None:
        self._success = False
        self._error_type = type(exc).__name__

    def add_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def build_record(self) -> ModelUsageRecord:
        elapsed = (time.monotonic() - self._start) * 1000
        return ModelUsageRecord(
            record_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            debate_id=self._debate_id,
            session_id=self._session_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            provider=self._provider,
            model=self._model,
            model_version=self._model_version,
            operation=self._operation,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            tokens_cached=self._tokens_cached,
            latency_ms=round(elapsed, 2),
            cost_usd=self._cost_usd,
            success=self._success,
            error_type=self._error_type,
            metadata=self._metadata,
        )


# ---------------------------------------------------------------------------
# Aggregator — roll up stored records by various dimensions
# ---------------------------------------------------------------------------


@dataclass
class UsageAggregate:
    """Aggregated model usage metrics."""

    group_key: str
    record_count: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tokens_cached: int = 0
    total_cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    error_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.record_count if self.record_count else 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out

    @property
    def error_rate(self) -> float:
        return self.error_count / self.record_count if self.record_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "record_count": self.record_count,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens_cached": self.total_tokens_cached,
            "total_tokens": self.total_tokens,
            "total_cost_usd": str(self.total_cost_usd),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
        }


class ModelUsageAggregator:
    """Aggregates model usage records by various dimensions."""

    @staticmethod
    def _accumulate(agg: UsageAggregate, rec: ModelUsageRecord) -> None:
        agg.record_count += 1
        agg.total_tokens_in += rec.tokens_in
        agg.total_tokens_out += rec.tokens_out
        agg.total_tokens_cached += rec.tokens_cached
        agg.total_cost_usd += Decimal(rec.cost_usd)
        agg.total_latency_ms += rec.latency_ms
        agg.min_latency_ms = min(agg.min_latency_ms, rec.latency_ms)
        agg.max_latency_ms = max(agg.max_latency_ms, rec.latency_ms)
        if not rec.success:
            agg.error_count += 1

    @classmethod
    def by_model(cls, records: list[ModelUsageRecord]) -> dict[str, UsageAggregate]:
        """Aggregate records grouped by model name."""
        groups: dict[str, UsageAggregate] = {}
        for rec in records:
            key = f"{rec.provider}/{rec.model}"
            if key not in groups:
                groups[key] = UsageAggregate(group_key=key)
            cls._accumulate(groups[key], rec)
        return groups

    @classmethod
    def by_debate(cls, records: list[ModelUsageRecord]) -> dict[str, UsageAggregate]:
        """Aggregate records grouped by debate_id."""
        groups: dict[str, UsageAggregate] = {}
        for rec in records:
            if rec.debate_id not in groups:
                groups[rec.debate_id] = UsageAggregate(group_key=rec.debate_id)
            cls._accumulate(groups[rec.debate_id], rec)
        return groups

    @classmethod
    def by_agent(cls, records: list[ModelUsageRecord]) -> dict[str, UsageAggregate]:
        """Aggregate records grouped by agent_name."""
        groups: dict[str, UsageAggregate] = {}
        for rec in records:
            if rec.agent_name not in groups:
                groups[rec.agent_name] = UsageAggregate(group_key=rec.agent_name)
            cls._accumulate(groups[rec.agent_name], rec)
        return groups

    @classmethod
    def by_operation(cls, records: list[ModelUsageRecord]) -> dict[str, UsageAggregate]:
        """Aggregate records grouped by operation type."""
        groups: dict[str, UsageAggregate] = {}
        for rec in records:
            if rec.operation not in groups:
                groups[rec.operation] = UsageAggregate(group_key=rec.operation)
            cls._accumulate(groups[rec.operation], rec)
        return groups

    @classmethod
    def total(cls, records: list[ModelUsageRecord]) -> UsageAggregate:
        """Single aggregate across all records."""
        agg = UsageAggregate(group_key="total")
        for rec in records:
            cls._accumulate(agg, rec)
        return agg


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def track_model_call(
    collector: ModelUsageCollector,
    *,
    debate_id: str,
    session_id: str,
    agent_id: str,
    agent_name: str,
    provider: str,
    model: str,
    operation: str,
    model_version: str = "",
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[_TrackingContext]:
    """Convenience wrapper around ``collector.track_call``."""
    return collector.track_call(
        debate_id=debate_id,
        session_id=session_id,
        agent_id=agent_id,
        agent_name=agent_name,
        provider=provider,
        model=model,
        operation=operation,
        model_version=model_version,
        metadata=metadata,
    )
