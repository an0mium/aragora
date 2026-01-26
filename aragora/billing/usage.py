"""
Usage Tracking System.

Tracks debates, tokens, and costs per user/organization for billing purposes.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class UsageEventType(Enum):
    """Types of usage events."""

    DEBATE = "debate"
    API_CALL = "api_call"
    STORAGE = "storage"
    AGENT_CALL = "agent_call"


# Provider pricing per 1M tokens (as of Jan 2026)
PROVIDER_PRICING: dict[str, dict[str, Decimal]] = {
    "anthropic": {
        "claude-opus-4": Decimal("15.00"),  # Input
        "claude-opus-4-output": Decimal("75.00"),
        "claude-sonnet-4": Decimal("3.00"),
        "claude-sonnet-4-output": Decimal("15.00"),
    },
    "openai": {
        "gpt-4o": Decimal("2.50"),
        "gpt-4o-output": Decimal("10.00"),
        "gpt-4o-mini": Decimal("0.15"),
        "gpt-4o-mini-output": Decimal("0.60"),
    },
    "google": {
        "gemini-pro": Decimal("1.25"),
        "gemini-pro-output": Decimal("5.00"),
    },
    "deepseek": {
        "deepseek-v3": Decimal("0.14"),
        "deepseek-v3-output": Decimal("0.28"),
    },
    "openrouter": {
        "default": Decimal("2.00"),
        "default-output": Decimal("8.00"),
    },
}


def calculate_token_cost(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> Decimal:
    """
    Calculate cost for token usage.

    Args:
        provider: Provider name (anthropic, openai, etc.)
        model: Model name
        tokens_in: Input tokens
        tokens_out: Output tokens

    Returns:
        Cost in USD
    """
    provider_prices = PROVIDER_PRICING.get(provider, PROVIDER_PRICING["openrouter"])

    # Get input price
    input_key = model if model in provider_prices else "default"
    input_price = provider_prices.get(input_key, Decimal("2.00"))

    # Get output price
    output_key = f"{model}-output" if f"{model}-output" in provider_prices else "default-output"
    output_price = provider_prices.get(output_key, Decimal("8.00"))

    # Calculate cost (prices are per 1M tokens)
    input_cost = (Decimal(tokens_in) / Decimal("1000000")) * input_price
    output_cost = (Decimal(tokens_out) / Decimal("1000000")) * output_price

    return input_cost + output_cost


@dataclass
class UsageEvent:
    """A single usage event."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    org_id: str = ""
    event_type: UsageEventType = UsageEventType.DEBATE
    debate_id: Optional[str] = None

    # Token usage
    tokens_in: int = 0
    tokens_out: int = 0

    # Provider info
    provider: str = ""
    model: str = ""

    # Cost
    cost_usd: Decimal = Decimal("0")

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def calculate_cost(self) -> Decimal:
        """Calculate and set cost based on tokens."""
        self.cost_usd = calculate_token_cost(
            self.provider, self.model, self.tokens_in, self.tokens_out
        )
        return self.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "event_type": self.event_type.value,
            "debate_id": self.debate_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "provider": self.provider,
            "model": self.model,
            "cost_usd": str(self.cost_usd),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageEvent":
        """Create from dictionary."""
        event = cls(
            id=data.get("id", str(uuid4())),
            user_id=data.get("user_id", ""),
            org_id=data.get("org_id", ""),
            event_type=UsageEventType(data.get("event_type", "debate")),
            debate_id=data.get("debate_id"),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            cost_usd=Decimal(data.get("cost_usd", "0")),
            metadata=data.get("metadata", {}),
        )
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                event.created_at = datetime.fromisoformat(data["created_at"])
            else:
                event.created_at = data["created_at"]
        return event


@dataclass
class UsageSummary:
    """Summary of usage for a period."""

    org_id: str
    period_start: datetime
    period_end: datetime

    # Counts
    total_debates: int = 0
    total_api_calls: int = 0
    total_agent_calls: int = 0

    # Tokens
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    # Cost
    total_cost_usd: Decimal = Decimal("0")

    # Breakdowns
    cost_by_provider: dict[str, Decimal] = field(default_factory=dict)
    debates_by_day: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_id": self.org_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_debates": self.total_debates,
            "total_api_calls": self.total_api_calls,
            "total_agent_calls": self.total_agent_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_cost_usd": str(self.total_cost_usd),
            "cost_by_provider": {k: str(v) for k, v in self.cost_by_provider.items()},
            "debates_by_day": self.debates_by_day,
        }


class UsageTracker:
    """
    Tracks and stores usage events.

    Provides methods for recording usage and generating summaries.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize usage tracker.

        Args:
            db_path: Path to SQLite database (default: .nomic/usage.db)
        """
        if db_path is None:
            db_path = Path(".nomic/usage.db")
        self.db_path = db_path
        self._ensure_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Create database schema if not exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    debate_id TEXT,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    provider TEXT,
                    model TEXT,
                    cost_usd TEXT DEFAULT '0',
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_usage_org_created
                    ON usage_events(org_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_usage_user_created
                    ON usage_events(user_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_usage_debate
                    ON usage_events(debate_id);
            """)
            conn.commit()

    def record(self, event: UsageEvent) -> None:
        """
        Record a usage event.

        Args:
            event: Usage event to record
        """
        import json

        # Calculate cost if not set
        if event.cost_usd == Decimal("0") and (event.tokens_in > 0 or event.tokens_out > 0):
            event.calculate_cost()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO usage_events
                (id, user_id, org_id, event_type, debate_id, tokens_in, tokens_out,
                 provider, model, cost_usd, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.user_id,
                    event.org_id,
                    event.event_type.value,
                    event.debate_id,
                    event.tokens_in,
                    event.tokens_out,
                    event.provider,
                    event.model,
                    str(event.cost_usd),
                    json.dumps(event.metadata),
                    event.created_at.isoformat(),
                ),
            )
            conn.commit()

        logger.debug(
            f"usage_recorded org={event.org_id} type={event.event_type.value} "
            f"tokens={event.tokens_in + event.tokens_out} cost=${event.cost_usd:.4f}"
        )

    def record_debate(
        self,
        user_id: str,
        org_id: str,
        debate_id: str,
        tokens_in: int,
        tokens_out: int,
        provider: str,
        model: str,
        metadata: Optional[dict] = None,
    ) -> UsageEvent:
        """
        Record a debate usage event.

        Args:
            user_id: User who initiated the debate
            org_id: Organization ID
            debate_id: Debate ID
            tokens_in: Input tokens used
            tokens_out: Output tokens generated
            provider: LLM provider
            model: Model used
            metadata: Additional metadata

        Returns:
            Created usage event
        """
        event = UsageEvent(
            user_id=user_id,
            org_id=org_id,
            event_type=UsageEventType.DEBATE,
            debate_id=debate_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            provider=provider,
            model=model,
            metadata=metadata or {},
        )
        event.calculate_cost()
        self.record(event)
        return event

    def record_agent_call(
        self,
        user_id: str,
        org_id: str,
        debate_id: Optional[str],
        agent_name: str,
        tokens_in: int,
        tokens_out: int,
        provider: str,
        model: str,
    ) -> UsageEvent:
        """
        Record an individual agent call.

        Args:
            user_id: User ID
            org_id: Organization ID
            debate_id: Associated debate ID (if any)
            agent_name: Name of the agent
            tokens_in: Input tokens
            tokens_out: Output tokens
            provider: LLM provider
            model: Model used

        Returns:
            Created usage event
        """
        event = UsageEvent(
            user_id=user_id,
            org_id=org_id,
            event_type=UsageEventType.AGENT_CALL,
            debate_id=debate_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            provider=provider,
            model=model,
            metadata={"agent": agent_name},
        )
        event.calculate_cost()
        self.record(event)
        return event

    def get_summary(
        self,
        org_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> UsageSummary:
        """
        Get usage summary for an organization.

        Args:
            org_id: Organization ID
            period_start: Start of period (default: 30 days ago)
            period_end: End of period (default: now)

        Returns:
            Usage summary
        """

        if period_end is None:
            period_end = datetime.now(timezone.utc)
        if period_start is None:
            period_start = period_end - timedelta(days=30)

        summary = UsageSummary(
            org_id=org_id,
            period_start=period_start,
            period_end=period_end,
        )

        with self._connection() as conn:
            # Get aggregate stats
            row = conn.execute(
                """
                SELECT
                    COUNT(CASE WHEN event_type = 'debate' THEN 1 END) as debates,
                    COUNT(CASE WHEN event_type = 'api_call' THEN 1 END) as api_calls,
                    COUNT(CASE WHEN event_type = 'agent_call' THEN 1 END) as agent_calls,
                    COALESCE(SUM(tokens_in), 0) as tokens_in,
                    COALESCE(SUM(tokens_out), 0) as tokens_out,
                    COALESCE(SUM(CAST(cost_usd AS REAL)), 0) as total_cost
                FROM usage_events
                WHERE org_id = ?
                    AND created_at >= ?
                    AND created_at <= ?
                """,
                (org_id, period_start.isoformat(), period_end.isoformat()),
            ).fetchone()

            if row:
                summary.total_debates = row["debates"]
                summary.total_api_calls = row["api_calls"]
                summary.total_agent_calls = row["agent_calls"]
                summary.total_tokens_in = row["tokens_in"]
                summary.total_tokens_out = row["tokens_out"]
                summary.total_cost_usd = Decimal(str(row["total_cost"]))

            # Get cost by provider
            rows = conn.execute(
                """
                SELECT provider, SUM(CAST(cost_usd AS REAL)) as cost
                FROM usage_events
                WHERE org_id = ?
                    AND created_at >= ?
                    AND created_at <= ?
                GROUP BY provider
                """,
                (org_id, period_start.isoformat(), period_end.isoformat()),
            ).fetchall()

            for row in rows:
                if row["provider"]:
                    summary.cost_by_provider[row["provider"]] = Decimal(str(row["cost"]))

            # Get debates by day
            rows = conn.execute(
                """
                SELECT DATE(created_at) as day, COUNT(*) as count
                FROM usage_events
                WHERE org_id = ?
                    AND event_type = 'debate'
                    AND created_at >= ?
                    AND created_at <= ?
                GROUP BY DATE(created_at)
                ORDER BY day
                """,
                (org_id, period_start.isoformat(), period_end.isoformat()),
            ).fetchall()

            for row in rows:
                summary.debates_by_day[row["day"]] = row["count"]

        return summary

    def get_user_usage(
        self,
        user_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get usage summary for a user.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            Usage statistics
        """
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=days)

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(CASE WHEN event_type = 'debate' THEN 1 END) as debates,
                    COALESCE(SUM(tokens_in + tokens_out), 0) as total_tokens,
                    COALESCE(SUM(CAST(cost_usd AS REAL)), 0) as total_cost
                FROM usage_events
                WHERE user_id = ?
                    AND created_at >= ?
                """,
                (user_id, period_start.isoformat()),
            ).fetchone()

            return {
                "user_id": user_id,
                "period_days": days,
                "debates": row["debates"] if row else 0,
                "total_tokens": row["total_tokens"] if row else 0,
                "total_cost_usd": str(Decimal(str(row["total_cost"])) if row else Decimal("0")),
            }

    def get_debate_cost(self, debate_id: str) -> Decimal:
        """
        Get total cost for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Total cost in USD
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(CAST(cost_usd AS REAL)), 0) as cost
                FROM usage_events
                WHERE debate_id = ?
                """,
                (debate_id,),
            ).fetchone()

            return Decimal(str(row["cost"])) if row else Decimal("0")

    def count_debates_this_month(self, org_id: str) -> int:
        """
        Count debates created this billing month.

        Args:
            org_id: Organization ID

        Returns:
            Number of debates
        """
        # Use first of current month as start
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM usage_events
                WHERE org_id = ?
                    AND event_type = 'debate'
                    AND created_at >= ?
                """,
                (org_id, month_start.isoformat()),
            ).fetchone()

            return row["count"] if row else 0


__all__ = [
    "UsageEventType",
    "UsageEvent",
    "UsageSummary",
    "UsageTracker",
    "calculate_token_cost",
    "PROVIDER_PRICING",
]
