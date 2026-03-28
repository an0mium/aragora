"""
Telemetry data models for AI model usage metrics.

Defines structured types for tracking tokens, latency, model version,
and cost per request — the building blocks for post-debate receipt
analytics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4


class RequestStatus(Enum):
    """Outcome status for a model request."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CANCELLED = "cancelled"


class TokenDirection(Enum):
    """Direction of token flow."""

    INPUT = "input"
    OUTPUT = "output"
    CACHED = "cached"


@dataclass(frozen=True)
class TokenCount:
    """Immutable token counts for a single request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cached_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenCount:
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cached_tokens=int(data.get("cached_tokens", 0)),
        )


@dataclass
class ModelUsageRecord:
    """
    A single AI model invocation record.

    Captures everything needed to attribute cost and latency to a debate,
    agent, and provider — the atomic unit of the analytics store.
    """

    id: str = field(default_factory=lambda: str(uuid4()))

    # Context
    debate_id: str = ""
    workspace_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    round_number: int | None = None

    # Model identification
    provider: str = ""
    model: str = ""
    model_version: str | None = None

    # Token counts
    tokens: TokenCount = field(default_factory=TokenCount)

    # Latency
    latency_ms: float = 0.0

    # Cost (Decimal for financial precision)
    cost_usd: Decimal = Decimal("0")

    # Status
    status: RequestStatus = RequestStatus.SUCCESS
    error_message: str | None = None

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Content hash for deduplication."""
        blob = (
            f"{self.debate_id}:{self.agent_id}:{self.provider}:{self.model}"
            f":{self.timestamp.isoformat()}:{self.tokens.total}"
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "debate_id": self.debate_id,
            "workspace_id": self.workspace_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "round_number": self.round_number,
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "input_tokens": self.tokens.input_tokens,
            "output_tokens": self.tokens.output_tokens,
            "cached_tokens": self.tokens.cached_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": str(self.cost_usd),
            "status": self.status.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelUsageRecord:
        tokens = TokenCount(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cached_tokens=int(data.get("cached_tokens", 0)),
        )
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now(timezone.utc)

        meta = data.get("metadata", {})
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}

        return cls(
            id=data.get("id", str(uuid4())),
            debate_id=data.get("debate_id", ""),
            workspace_id=data.get("workspace_id", ""),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            round_number=data.get("round_number"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            model_version=data.get("model_version"),
            tokens=tokens,
            latency_ms=float(data.get("latency_ms", 0.0)),
            cost_usd=Decimal(str(data.get("cost_usd", "0"))),
            status=RequestStatus(data.get("status", "success")),
            error_message=data.get("error_message"),
            timestamp=ts,
            metadata=meta,
        )


@dataclass
class DebateUsageSummary:
    """
    Aggregated usage for an entire debate — the receipt-level view.

    Built by the analytics store from individual ModelUsageRecord rows.
    """

    debate_id: str
    workspace_id: str = ""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0

    total_cost_usd: Decimal = Decimal("0")

    # Latency stats (milliseconds)
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Breakdowns
    cost_by_provider: dict[str, Decimal] = field(default_factory=dict)
    cost_by_model: dict[str, Decimal] = field(default_factory=dict)
    tokens_by_agent: dict[str, int] = field(default_factory=dict)
    requests_by_round: dict[int, int] = field(default_factory=dict)

    started_at: datetime | None = None
    ended_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "workspace_id": self.workspace_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_usd": str(self.total_cost_usd),
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "cost_by_provider": {k: str(v) for k, v in self.cost_by_provider.items()},
            "cost_by_model": {k: str(v) for k, v in self.cost_by_model.items()},
            "tokens_by_agent": self.tokens_by_agent,
            "requests_by_round": {str(k): v for k, v in self.requests_by_round.items()},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }
