"""
Backend telemetry pipeline for AI model usage metrics.

Captures per-request AI model usage metrics during debate sessions:
- Token counts (input, output, cached)
- Latency per call
- Model version and provider
- Computed cost

Accumulates records per-debate and produces summaries suitable for
populating ``DecisionReceipt.cost_summary``.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelUsageRecord:
    """Immutable record of a single AI model API call."""

    record_id: str = field(default_factory=lambda: str(uuid4()))
    debate_id: str = ""
    agent_id: str = ""
    provider: str = ""
    model: str = ""
    model_version: str = ""

    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0

    latency_ms: float = 0.0
    cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))

    operation: str = ""  # e.g. "proposal", "critique", "revision", "vote"
    round_number: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class ModelUsageCollector:
    """Thread-safe collector that accumulates per-request model usage records.

    Records are indexed by debate_id for efficient per-debate summarisation.
    The collector is designed to be used as a singleton alongside the existing
    ``TelemetryCollector`` — it focuses exclusively on LLM call metrics.
    """

    def __init__(self, *, max_records_per_debate: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, list[ModelUsageRecord]] = defaultdict(list)
        self._max_records_per_debate = max_records_per_debate

    # -- Recording --------------------------------------------------------

    def record(self, usage: ModelUsageRecord) -> None:
        """Record a model usage event (thread-safe)."""
        key = usage.debate_id or "__global__"
        with self._lock:
            bucket = self._records[key]
            if len(bucket) >= self._max_records_per_debate:
                logger.warning(
                    "model_usage_limit_reached debate_id=%s limit=%d",
                    key,
                    self._max_records_per_debate,
                )
                return
            bucket.append(usage)

    # -- Querying ---------------------------------------------------------

    def get_records(self, debate_id: str) -> list[ModelUsageRecord]:
        """Return a copy of all records for a given debate."""
        with self._lock:
            return list(self._records.get(debate_id, []))

    def get_all_debate_ids(self) -> list[str]:
        """Return debate ids that have at least one record."""
        with self._lock:
            return [k for k in self._records if k != "__global__"]

    # -- Summarisation for receipts ---------------------------------------

    def summarise_debate(self, debate_id: str) -> dict[str, Any]:
        """Produce a cost summary dict suitable for ``DecisionReceipt.cost_summary``.

        Returns a dict with:
        - total_cost_usd: str (decimal string)
        - total_tokens_in / total_tokens_out / total_tokens_cached: int
        - total_latency_ms: float
        - call_count: int
        - by_model: dict mapping "provider/model" → per-model breakdown
        - by_agent: dict mapping agent_id → per-agent breakdown
        - by_operation: dict mapping operation → per-operation breakdown
        """
        records = self.get_records(debate_id)
        if not records:
            return {
                "total_cost_usd": "0",
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens_cached": 0,
                "total_latency_ms": 0.0,
                "call_count": 0,
                "by_model": {},
                "by_agent": {},
                "by_operation": {},
            }

        total_cost = Decimal("0")
        total_in = 0
        total_out = 0
        total_cached = 0
        total_latency = 0.0

        by_model: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "cost_usd": Decimal("0"),
                "tokens_in": 0,
                "tokens_out": 0,
                "call_count": 0,
                "latency_ms": 0.0,
                "model_version": "",
            }
        )
        by_agent: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "cost_usd": Decimal("0"),
                "tokens_in": 0,
                "tokens_out": 0,
                "call_count": 0,
                "latency_ms": 0.0,
            }
        )
        by_operation: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "cost_usd": Decimal("0"),
                "tokens_in": 0,
                "tokens_out": 0,
                "call_count": 0,
                "latency_ms": 0.0,
            }
        )

        for r in records:
            total_cost += r.cost_usd
            total_in += r.tokens_in
            total_out += r.tokens_out
            total_cached += r.tokens_cached
            total_latency += r.latency_ms

            model_key = f"{r.provider}/{r.model}" if r.provider else r.model
            m = by_model[model_key]
            m["cost_usd"] += r.cost_usd
            m["tokens_in"] += r.tokens_in
            m["tokens_out"] += r.tokens_out
            m["call_count"] += 1
            m["latency_ms"] += r.latency_ms
            if r.model_version:
                m["model_version"] = r.model_version

            a = by_agent[r.agent_id]
            a["cost_usd"] += r.cost_usd
            a["tokens_in"] += r.tokens_in
            a["tokens_out"] += r.tokens_out
            a["call_count"] += 1
            a["latency_ms"] += r.latency_ms

            if r.operation:
                o = by_operation[r.operation]
                o["cost_usd"] += r.cost_usd
                o["tokens_in"] += r.tokens_in
                o["tokens_out"] += r.tokens_out
                o["call_count"] += 1
                o["latency_ms"] += r.latency_ms

        def _serialise_breakdown(d: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for key, val in d.items():
                out[key] = {k: (str(v) if isinstance(v, Decimal) else v) for k, v in val.items()}
            return out

        return {
            "total_cost_usd": str(total_cost),
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_tokens_cached": total_cached,
            "total_latency_ms": round(total_latency, 2),
            "call_count": len(records),
            "by_model": _serialise_breakdown(by_model),
            "by_agent": _serialise_breakdown(by_agent),
            "by_operation": _serialise_breakdown(by_operation),
        }

    # -- Lifecycle --------------------------------------------------------

    def clear_debate(self, debate_id: str) -> int:
        """Remove all records for a debate, returning count removed."""
        with self._lock:
            removed = self._records.pop(debate_id, [])
            return len(removed)

    def clear_all(self) -> None:
        """Remove all records."""
        with self._lock:
            self._records.clear()


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_default_collector: ModelUsageCollector | None = None
_singleton_lock = threading.Lock()


def get_model_usage_collector() -> ModelUsageCollector:
    """Return the process-wide ``ModelUsageCollector`` singleton."""
    global _default_collector
    if _default_collector is None:
        with _singleton_lock:
            if _default_collector is None:
                _default_collector = ModelUsageCollector()
    return _default_collector


def set_model_usage_collector(collector: ModelUsageCollector) -> None:
    """Replace the process-wide singleton (useful in tests)."""
    global _default_collector
    with _singleton_lock:
        _default_collector = collector


def record_model_usage(
    *,
    debate_id: str = "",
    agent_id: str = "",
    provider: str = "",
    model: str = "",
    model_version: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    tokens_cached: int = 0,
    latency_ms: float = 0.0,
    cost_usd: Decimal | float | str = 0,
    operation: str = "",
    round_number: int = 0,
    metadata: dict[str, Any] | None = None,
) -> ModelUsageRecord:
    """Convenience function: create a record and add it to the default collector.

    Returns the created ``ModelUsageRecord`` for caller inspection.
    """
    record = ModelUsageRecord(
        debate_id=debate_id,
        agent_id=agent_id,
        provider=provider,
        model=model,
        model_version=model_version,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_cached=tokens_cached,
        latency_ms=latency_ms,
        cost_usd=Decimal(str(cost_usd)),
        operation=operation,
        round_number=round_number,
        metadata=metadata or {},
    )
    get_model_usage_collector().record(record)
    return record
