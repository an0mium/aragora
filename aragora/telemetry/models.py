"""
Data models for AI model usage telemetry.

Tracks per-request metrics (tokens, latency, cost, model version) and
associates them with debate sessions for analytics and cost attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4


class UsageOperation(str, Enum):
    """Types of operations that consume model tokens."""

    DEBATE_ROUND = "debate_round"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    REVISION = "revision"
    CONSENSUS_CHECK = "consensus_check"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    EMBEDDING = "embedding"
    MODERATION = "moderation"
    OTHER = "other"


class AggregationInterval(str, Enum):
    """Time intervals for aggregating usage data."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ModelUsageRecord:
    """A single AI model invocation record with full telemetry.

    Captures everything needed to attribute cost, measure performance,
    and audit model usage within a debate session.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    debate_id: str | None = None
    session_id: str | None = None
    workspace_id: str = ""
    agent_id: str = ""
    agent_name: str = ""

    # Model info
    provider: str = ""
    model: str = ""
    model_version: str = ""

    # Token counts
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    tokens_total: int = 0

    # Performance
    latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None

    # Cost
    cost_usd: Decimal = Decimal("0")

    # Context
    operation: UsageOperation = UsageOperation.OTHER
    round_number: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Extensible metadata (e.g., temperature, max_tokens, stop_reason)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens_total == 0:
            self.tokens_total = self.tokens_input + self.tokens_output

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "id": self.id,
            "debate_id": self.debate_id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_cached": self.tokens_cached,
            "tokens_total": self.tokens_total,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "cost_usd": str(self.cost_usd),
            "operation": self.operation.value,
            "round_number": self.round_number,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelUsageRecord:
        """Deserialize from a dictionary."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now(timezone.utc)

        op = data.get("operation", "other")
        if isinstance(op, str):
            try:
                op = UsageOperation(op)
            except ValueError:
                op = UsageOperation.OTHER

        return cls(
            id=data.get("id", str(uuid4())),
            debate_id=data.get("debate_id"),
            session_id=data.get("session_id"),
            workspace_id=data.get("workspace_id", ""),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            model_version=data.get("model_version", ""),
            tokens_input=data.get("tokens_input", 0),
            tokens_output=data.get("tokens_output", 0),
            tokens_cached=data.get("tokens_cached", 0),
            tokens_total=data.get("tokens_total", 0),
            latency_ms=data.get("latency_ms", 0.0),
            time_to_first_token_ms=data.get("time_to_first_token_ms"),
            cost_usd=Decimal(str(data.get("cost_usd", "0"))),
            operation=op,
            round_number=data.get("round_number"),
            timestamp=ts,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_row(cls, row: tuple) -> ModelUsageRecord:
        """Create from a database row (ordered by COLUMNS)."""
        op = row[15]
        try:
            op = UsageOperation(op)
        except (ValueError, KeyError):
            op = UsageOperation.OTHER

        ts = row[17]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        return cls(
            id=row[0],
            debate_id=row[1],
            session_id=row[2],
            workspace_id=row[3] or "",
            agent_id=row[4] or "",
            agent_name=row[5] or "",
            provider=row[6] or "",
            model=row[7] or "",
            model_version=row[8] or "",
            tokens_input=row[9] or 0,
            tokens_output=row[10] or 0,
            tokens_cached=row[11] or 0,
            tokens_total=row[12] or 0,
            latency_ms=row[13] or 0.0,
            time_to_first_token_ms=row[14],
            cost_usd=Decimal(str(row[16] or "0")),
            operation=op,
            round_number=row[18],
            timestamp=ts,
            metadata={},
        )


# Column order matching from_row
COLUMNS = (
    "id",
    "debate_id",
    "session_id",
    "workspace_id",
    "agent_id",
    "agent_name",
    "provider",
    "model",
    "model_version",
    "tokens_input",
    "tokens_output",
    "tokens_cached",
    "tokens_total",
    "latency_ms",
    "time_to_first_token_ms",
    "operation",
    "cost_usd",
    "timestamp",
    "round_number",
)


@dataclass
class SessionUsageSummary:
    """Aggregated usage metrics for a single debate session."""

    debate_id: str
    session_id: str | None = None
    total_requests: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_tokens: int = 0
    total_cost_usd: Decimal = Decimal("0")
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    models_used: list[str] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    first_request_at: datetime | None = None
    last_request_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "debate_id": self.debate_id,
            "session_id": self.session_id,
            "total_requests": self.total_requests,
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_tokens": self.total_tokens,
            "total_cost_usd": str(self.total_cost_usd),
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "models_used": self.models_used,
            "providers_used": self.providers_used,
            "duration_seconds": self.duration_seconds,
            "first_request_at": self.first_request_at.isoformat()
            if self.first_request_at
            else None,
            "last_request_at": self.last_request_at.isoformat() if self.last_request_at else None,
        }


@dataclass
class UsageQuery:
    """Query parameters for filtering usage records."""

    debate_id: str | None = None
    session_id: str | None = None
    workspace_id: str | None = None
    agent_id: str | None = None
    provider: str | None = None
    model: str | None = None
    operation: UsageOperation | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int = 1000
    offset: int = 0

    def to_where_clause(self) -> tuple[str, list[Any]]:
        """Build a SQL WHERE clause from the query parameters.

        Returns:
            Tuple of (where_clause, params) for SQL query construction.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if self.debate_id is not None:
            conditions.append("debate_id = ?")
            params.append(self.debate_id)
        if self.session_id is not None:
            conditions.append("session_id = ?")
            params.append(self.session_id)
        if self.workspace_id is not None:
            conditions.append("workspace_id = ?")
            params.append(self.workspace_id)
        if self.agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(self.agent_id)
        if self.provider is not None:
            conditions.append("provider = ?")
            params.append(self.provider)
        if self.model is not None:
            conditions.append("model = ?")
            params.append(self.model)
        if self.operation is not None:
            conditions.append("operation = ?")
            params.append(self.operation.value)
        if self.start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(self.start_time.isoformat())
        if self.end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(self.end_time.isoformat())

        where = " AND ".join(conditions) if conditions else "1=1"
        return where, params


@dataclass
class ModelUsageAggregate:
    """Aggregated usage grouped by a dimension (model, provider, agent, etc.)."""

    group_key: str
    group_value: str
    total_requests: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_tokens: int = 0
    total_cost_usd: Decimal = Decimal("0")
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "group_key": self.group_key,
            "group_value": self.group_value,
            "total_requests": self.total_requests,
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_tokens": self.total_tokens,
            "total_cost_usd": str(self.total_cost_usd),
            "avg_latency_ms": self.avg_latency_ms,
        }
