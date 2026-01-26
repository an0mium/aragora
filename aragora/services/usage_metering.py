"""
Usage-Based Billing Metering Service for ENTERPRISE_PLUS Tier.

Provides comprehensive token-level metering with:
- Per-model, per-provider cost tracking
- Debate and API call metering
- Hourly aggregation for efficient storage
- Integration with billing infrastructure

Phase 4.3 Implementation as per development roadmap.

Usage:
    from aragora.services.usage_metering import UsageMeter, get_usage_meter

    meter = get_usage_meter()

    # Record token usage
    await meter.record_token_usage(
        org_id="org_123",
        input_tokens=1000,
        output_tokens=500,
        model="claude-opus-4",
    )

    # Record debate usage
    await meter.record_debate_usage(
        org_id="org_123",
        debate_id="debate_456",
        agent_count=3,
    )

    # Get usage summary
    summary = await meter.get_usage_summary(org_id="org_123", period="month")
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_METERING_DB = Path(".nomic/usage_metering.db")


class MeteringPeriod(Enum):
    """Billing period types for usage queries."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class UsageType(Enum):
    """Types of metered usage events."""

    TOKEN = "token"
    DEBATE = "debate"
    API_CALL = "api_call"
    STORAGE = "storage"
    CONNECTOR = "connector"


# Provider pricing per 1M tokens (as of Jan 2026)
# Aligned with aragora.billing.usage.PROVIDER_PRICING
MODEL_PRICING: Dict[str, Dict[str, Decimal]] = {
    "anthropic": {
        "claude-opus-4": Decimal("15.00"),
        "claude-opus-4-output": Decimal("75.00"),
        "claude-sonnet-4": Decimal("3.00"),
        "claude-sonnet-4-output": Decimal("15.00"),
        "claude-haiku-4": Decimal("0.25"),
        "claude-haiku-4-output": Decimal("1.25"),
        "default": Decimal("3.00"),
        "default-output": Decimal("15.00"),
    },
    "openai": {
        "gpt-4o": Decimal("2.50"),
        "gpt-4o-output": Decimal("10.00"),
        "gpt-4o-mini": Decimal("0.15"),
        "gpt-4o-mini-output": Decimal("0.60"),
        "o1": Decimal("15.00"),
        "o1-output": Decimal("60.00"),
        "default": Decimal("2.50"),
        "default-output": Decimal("10.00"),
    },
    "google": {
        "gemini-pro": Decimal("1.25"),
        "gemini-pro-output": Decimal("5.00"),
        "gemini-ultra": Decimal("10.00"),
        "gemini-ultra-output": Decimal("30.00"),
        "default": Decimal("1.25"),
        "default-output": Decimal("5.00"),
    },
    "deepseek": {
        "deepseek-v3": Decimal("0.14"),
        "deepseek-v3-output": Decimal("0.28"),
        "default": Decimal("0.14"),
        "default-output": Decimal("0.28"),
    },
    "xai": {
        "grok-2": Decimal("2.00"),
        "grok-2-output": Decimal("10.00"),
        "default": Decimal("2.00"),
        "default-output": Decimal("10.00"),
    },
    "mistral": {
        "mistral-large": Decimal("2.00"),
        "mistral-large-output": Decimal("6.00"),
        "codestral": Decimal("0.20"),
        "codestral-output": Decimal("0.60"),
        "default": Decimal("2.00"),
        "default-output": Decimal("6.00"),
    },
    "openrouter": {
        "default": Decimal("2.00"),
        "default-output": Decimal("8.00"),
    },
}

# Tier-based usage caps (monthly)
TIER_USAGE_CAPS: Dict[str, Dict[str, int]] = {
    "free": {
        "max_tokens": 100_000,
        "max_debates": 10,
        "max_api_calls": 100,
    },
    "starter": {
        "max_tokens": 1_000_000,
        "max_debates": 50,
        "max_api_calls": 1_000,
    },
    "professional": {
        "max_tokens": 10_000_000,
        "max_debates": 200,
        "max_api_calls": 10_000,
    },
    "enterprise": {
        "max_tokens": 100_000_000,
        "max_debates": 999_999,
        "max_api_calls": 999_999,
    },
    "enterprise_plus": {
        "max_tokens": 999_999_999,  # Effectively unlimited
        "max_debates": 999_999,
        "max_api_calls": 999_999,
    },
}


@dataclass
class TokenUsageRecord:
    """Record of token usage for a single API call."""

    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str = ""
    user_id: Optional[str] = None
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: Decimal = Decimal("0")
    output_cost: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    debate_id: Optional[str] = None
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": str(self.input_cost),
            "output_cost": str(self.output_cost),
            "total_cost": str(self.total_cost),
            "debate_id": self.debate_id,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DebateUsageRecord:
    """Record of debate usage."""

    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str = ""
    user_id: Optional[str] = None
    debate_id: str = ""
    agent_count: int = 0
    rounds: int = 0
    total_tokens: int = 0
    total_cost: Decimal = Decimal("0")
    duration_seconds: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "debate_id": self.debate_id,
            "agent_count": self.agent_count,
            "rounds": self.rounds,
            "total_tokens": self.total_tokens,
            "total_cost": str(self.total_cost),
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ApiCallRecord:
    """Record of API call usage."""

    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str = ""
    user_id: Optional[str] = None
    endpoint: str = ""
    method: str = "GET"
    status_code: int = 200
    response_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HourlyAggregate:
    """Hourly aggregation of usage for efficient storage."""

    org_id: str = ""
    hour: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    usage_type: UsageType = UsageType.TOKEN

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_cost: Decimal = Decimal("0")

    # Counts
    debate_count: int = 0
    api_call_count: int = 0

    # By model/provider breakdown
    tokens_by_model: Dict[str, int] = field(default_factory=dict)
    cost_by_model: Dict[str, Decimal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_id": self.org_id,
            "hour": self.hour.isoformat(),
            "usage_type": self.usage_type.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "token_cost": str(self.token_cost),
            "debate_count": self.debate_count,
            "api_call_count": self.api_call_count,
            "tokens_by_model": self.tokens_by_model,
            "cost_by_model": {k: str(v) for k, v in self.cost_by_model.items()},
        }


@dataclass
class UsageSummary:
    """Summary of usage for a billing period."""

    org_id: str
    period_start: datetime
    period_end: datetime
    period_type: MeteringPeriod = MeteringPeriod.MONTH

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_cost: Decimal = Decimal("0")

    # Counts
    debate_count: int = 0
    api_call_count: int = 0

    # Breakdowns
    tokens_by_model: Dict[str, int] = field(default_factory=dict)
    cost_by_model: Dict[str, Decimal] = field(default_factory=dict)
    tokens_by_provider: Dict[str, int] = field(default_factory=dict)
    cost_by_provider: Dict[str, Decimal] = field(default_factory=dict)
    cost_by_day: Dict[str, Decimal] = field(default_factory=dict)

    # Limits
    token_limit: int = 0
    debate_limit: int = 0
    api_call_limit: int = 0

    # Usage percentages
    token_usage_percent: float = 0.0
    debate_usage_percent: float = 0.0
    api_call_usage_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_id": self.org_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "period_type": self.period_type.value,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.total_tokens,
                "cost": str(self.token_cost),
            },
            "counts": {
                "debates": self.debate_count,
                "api_calls": self.api_call_count,
            },
            "by_model": {
                "tokens": self.tokens_by_model,
                "cost": {k: str(v) for k, v in self.cost_by_model.items()},
            },
            "by_provider": {
                "tokens": self.tokens_by_provider,
                "cost": {k: str(v) for k, v in self.cost_by_provider.items()},
            },
            "by_day": {k: str(v) for k, v in self.cost_by_day.items()},
            "limits": {
                "tokens": self.token_limit,
                "debates": self.debate_limit,
                "api_calls": self.api_call_limit,
            },
            "usage_percent": {
                "tokens": self.token_usage_percent,
                "debates": self.debate_usage_percent,
                "api_calls": self.api_call_usage_percent,
            },
        }


@dataclass
class UsageLimits:
    """Current usage limits and utilization."""

    org_id: str
    tier: str = "free"

    # Limits
    max_tokens: int = 0
    max_debates: int = 0
    max_api_calls: int = 0

    # Current usage
    tokens_used: int = 0
    debates_used: int = 0
    api_calls_used: int = 0

    # Percentages
    tokens_percent: float = 0.0
    debates_percent: float = 0.0
    api_calls_percent: float = 0.0

    # Flags
    tokens_exceeded: bool = False
    debates_exceeded: bool = False
    api_calls_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_id": self.org_id,
            "tier": self.tier,
            "limits": {
                "tokens": self.max_tokens,
                "debates": self.max_debates,
                "api_calls": self.max_api_calls,
            },
            "used": {
                "tokens": self.tokens_used,
                "debates": self.debates_used,
                "api_calls": self.api_calls_used,
            },
            "percent": {
                "tokens": self.tokens_percent,
                "debates": self.debates_percent,
                "api_calls": self.api_calls_percent,
            },
            "exceeded": {
                "tokens": self.tokens_exceeded,
                "debates": self.debates_exceeded,
                "api_calls": self.api_calls_exceeded,
            },
        }


@dataclass
class UsageBreakdown:
    """Detailed usage breakdown for billing."""

    org_id: str
    period_start: datetime
    period_end: datetime

    # Totals
    total_cost: Decimal = Decimal("0")
    total_tokens: int = 0
    total_debates: int = 0
    total_api_calls: int = 0

    # By model
    by_model: List[Dict[str, Any]] = field(default_factory=list)

    # By provider
    by_provider: List[Dict[str, Any]] = field(default_factory=list)

    # By day
    by_day: List[Dict[str, Any]] = field(default_factory=list)

    # By user (for multi-user organizations)
    by_user: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_id": self.org_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "totals": {
                "cost": str(self.total_cost),
                "tokens": self.total_tokens,
                "debates": self.total_debates,
                "api_calls": self.total_api_calls,
            },
            "by_model": self.by_model,
            "by_provider": self.by_provider,
            "by_day": self.by_day,
            "by_user": self.by_user,
        }


class UsageMeter:
    """
    Usage metering service for ENTERPRISE_PLUS tier.

    Provides comprehensive token-level tracking, hourly aggregation,
    and billing integration.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize usage meter.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or DEFAULT_METERING_DB
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

        # In-memory buffer for batching writes
        self._token_buffer: List[TokenUsageRecord] = []
        self._debate_buffer: List[DebateUsageRecord] = []
        self._api_buffer: List[ApiCallRecord] = []
        self._buffer_size = 50

    async def initialize(self) -> None:
        """Initialize database and schema."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        await self._init_schema()
        self._initialized = True
        logger.info(f"Usage metering initialized: {self.db_path}")

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self._conn.cursor()

        # Token usage records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                user_id TEXT,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                input_cost TEXT DEFAULT '0',
                output_cost TEXT DEFAULT '0',
                total_cost TEXT DEFAULT '0',
                debate_id TEXT,
                endpoint TEXT,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Debate usage records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS debate_usage (
                id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                user_id TEXT,
                debate_id TEXT NOT NULL,
                agent_count INTEGER DEFAULT 0,
                rounds INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_cost TEXT DEFAULT '0',
                duration_seconds INTEGER DEFAULT 0,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # API call records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                user_id TEXT,
                endpoint TEXT NOT NULL,
                method TEXT DEFAULT 'GET',
                status_code INTEGER DEFAULT 200,
                response_time_ms INTEGER DEFAULT 0,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Hourly aggregates for efficient queries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_aggregates (
                org_id TEXT NOT NULL,
                hour TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                token_cost TEXT DEFAULT '0',
                debate_count INTEGER DEFAULT 0,
                api_call_count INTEGER DEFAULT 0,
                tokens_by_model TEXT DEFAULT '{}',
                cost_by_model TEXT DEFAULT '{}',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (org_id, hour)
            )
        """)

        # Indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_org_time
            ON token_usage(org_id, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_model
            ON token_usage(provider, model)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_debate_usage_org_time
            ON debate_usage(org_id, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_usage_org_time
            ON api_usage(org_id, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hourly_agg_org_hour
            ON hourly_aggregates(org_id, hour)
        """)

        self._conn.commit()

    def _calculate_token_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate input and output costs for token usage.

        Args:
            provider: Provider name (anthropic, openai, etc.)
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Tuple of (input_cost, output_cost)
        """
        provider_lower = provider.lower()
        provider_prices = MODEL_PRICING.get(provider_lower, MODEL_PRICING["openrouter"])

        # Get input price
        input_key = model if model in provider_prices else "default"
        input_price = provider_prices.get(input_key, Decimal("2.00"))

        # Get output price
        output_key = f"{model}-output" if f"{model}-output" in provider_prices else "default-output"
        output_price = provider_prices.get(output_key, Decimal("8.00"))

        # Calculate costs (prices are per 1M tokens)
        input_cost = (Decimal(input_tokens) / Decimal("1000000")) * input_price
        output_cost = (Decimal(output_tokens) / Decimal("1000000")) * output_price

        return input_cost, output_cost

    async def record_token_usage(
        self,
        org_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        provider: str = "openrouter",
        user_id: Optional[str] = None,
        debate_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenUsageRecord:
        """
        Record token usage for billing.

        Args:
            org_id: Organization identifier
            input_tokens: Input token count
            output_tokens: Output token count
            model: Model name
            provider: Provider name
            user_id: Optional user identifier
            debate_id: Optional debate identifier
            endpoint: Optional endpoint that generated the usage
            metadata: Additional metadata

        Returns:
            Created token usage record
        """
        if not self._initialized:
            await self.initialize()

        # Calculate costs
        input_cost, output_cost = self._calculate_token_cost(
            provider, model, input_tokens, output_tokens
        )
        total_cost = input_cost + output_cost

        record = TokenUsageRecord(
            org_id=org_id,
            user_id=user_id,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            debate_id=debate_id,
            endpoint=endpoint,
            metadata=metadata or {},
        )

        async with self._lock:
            self._token_buffer.append(record)
            if len(self._token_buffer) >= self._buffer_size:
                await self._flush_token_buffer()

        # Update hourly aggregate
        await self._update_hourly_aggregate(
            org_id=org_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_cost=total_cost,
            model=f"{provider}/{model}",
        )

        logger.debug(
            f"Token usage recorded: org={org_id} model={model} "
            f"tokens={input_tokens + output_tokens} cost=${total_cost:.4f}"
        )

        return record

    async def record_debate_usage(
        self,
        org_id: str,
        debate_id: str,
        agent_count: int,
        rounds: int = 0,
        total_tokens: int = 0,
        total_cost: Optional[Decimal] = None,
        duration_seconds: int = 0,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DebateUsageRecord:
        """
        Record debate usage for billing.

        Args:
            org_id: Organization identifier
            debate_id: Debate identifier
            agent_count: Number of agents in debate
            rounds: Number of debate rounds
            total_tokens: Total tokens used in debate
            total_cost: Total cost (calculated if not provided)
            duration_seconds: Debate duration
            user_id: Optional user identifier
            metadata: Additional metadata

        Returns:
            Created debate usage record
        """
        if not self._initialized:
            await self.initialize()

        # Calculate cost if not provided
        if total_cost is None:
            # Estimate cost based on tokens
            _, output_cost = self._calculate_token_cost(
                "openrouter", "default", total_tokens // 2, total_tokens // 2
            )
            total_cost = output_cost * 2

        record = DebateUsageRecord(
            org_id=org_id,
            user_id=user_id,
            debate_id=debate_id,
            agent_count=agent_count,
            rounds=rounds,
            total_tokens=total_tokens,
            total_cost=total_cost,
            duration_seconds=duration_seconds,
            metadata=metadata or {},
        )

        async with self._lock:
            self._debate_buffer.append(record)
            if len(self._debate_buffer) >= self._buffer_size:
                await self._flush_debate_buffer()

        # Update hourly aggregate
        await self._update_hourly_aggregate(
            org_id=org_id,
            debate_count=1,
        )

        logger.debug(
            f"Debate usage recorded: org={org_id} debate={debate_id} "
            f"agents={agent_count} cost=${total_cost:.4f}"
        )

        return record

    async def record_api_call(
        self,
        org_id: str,
        endpoint: str,
        method: str = "GET",
        status_code: int = 200,
        response_time_ms: int = 0,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApiCallRecord:
        """
        Record API call for metering.

        Args:
            org_id: Organization identifier
            endpoint: API endpoint called
            method: HTTP method
            status_code: Response status code
            response_time_ms: Response time in milliseconds
            user_id: Optional user identifier
            metadata: Additional metadata

        Returns:
            Created API call record
        """
        if not self._initialized:
            await self.initialize()

        record = ApiCallRecord(
            org_id=org_id,
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            metadata=metadata or {},
        )

        async with self._lock:
            self._api_buffer.append(record)
            if len(self._api_buffer) >= self._buffer_size:
                await self._flush_api_buffer()

        # Update hourly aggregate
        await self._update_hourly_aggregate(
            org_id=org_id,
            api_call_count=1,
        )

        return record

    async def _update_hourly_aggregate(
        self,
        org_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        token_cost: Decimal = Decimal("0"),
        model: str = "",
        debate_count: int = 0,
        api_call_count: int = 0,
    ) -> None:
        """Update hourly aggregate with new usage."""
        now = datetime.now(timezone.utc)
        hour = now.replace(minute=0, second=0, microsecond=0)
        hour_str = hour.isoformat()

        cursor = self._conn.cursor()

        # Get existing aggregate
        cursor.execute(
            """
            SELECT * FROM hourly_aggregates
            WHERE org_id = ? AND hour = ?
            """,
            (org_id, hour_str),
        )
        row = cursor.fetchone()

        if row:
            # Update existing aggregate
            tokens_by_model = json.loads(row["tokens_by_model"] or "{}")
            cost_by_model = json.loads(row["cost_by_model"] or "{}")

            if model:
                tokens_by_model[model] = (
                    tokens_by_model.get(model, 0) + input_tokens + output_tokens
                )
                cost_by_model[model] = str(Decimal(cost_by_model.get(model, "0")) + token_cost)

            cursor.execute(
                """
                UPDATE hourly_aggregates
                SET input_tokens = input_tokens + ?,
                    output_tokens = output_tokens + ?,
                    total_tokens = total_tokens + ?,
                    token_cost = CAST((CAST(token_cost AS REAL) + ?) AS TEXT),
                    debate_count = debate_count + ?,
                    api_call_count = api_call_count + ?,
                    tokens_by_model = ?,
                    cost_by_model = ?,
                    updated_at = ?
                WHERE org_id = ? AND hour = ?
                """,
                (
                    input_tokens,
                    output_tokens,
                    input_tokens + output_tokens,
                    float(token_cost),
                    debate_count,
                    api_call_count,
                    json.dumps(tokens_by_model),
                    json.dumps(cost_by_model),
                    now.isoformat(),
                    org_id,
                    hour_str,
                ),
            )
        else:
            # Create new aggregate
            tokens_by_model = {}
            cost_by_model = {}
            if model:
                tokens_by_model[model] = input_tokens + output_tokens
                cost_by_model[model] = str(token_cost)

            cursor.execute(
                """
                INSERT INTO hourly_aggregates
                (org_id, hour, input_tokens, output_tokens, total_tokens,
                 token_cost, debate_count, api_call_count,
                 tokens_by_model, cost_by_model, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    org_id,
                    hour_str,
                    input_tokens,
                    output_tokens,
                    input_tokens + output_tokens,
                    str(token_cost),
                    debate_count,
                    api_call_count,
                    json.dumps(tokens_by_model),
                    json.dumps(cost_by_model),
                    now.isoformat(),
                ),
            )

        self._conn.commit()

    async def _flush_token_buffer(self) -> None:
        """Flush token usage buffer to database."""
        if not self._token_buffer:
            return

        records = self._token_buffer.copy()
        self._token_buffer.clear()

        cursor = self._conn.cursor()
        for record in records:
            cursor.execute(
                """
                INSERT INTO token_usage
                (id, org_id, user_id, model, provider, input_tokens, output_tokens,
                 total_tokens, input_cost, output_cost, total_cost,
                 debate_id, endpoint, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.org_id,
                    record.user_id,
                    record.model,
                    record.provider,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    str(record.input_cost),
                    str(record.output_cost),
                    str(record.total_cost),
                    record.debate_id,
                    record.endpoint,
                    json.dumps(record.metadata) if record.metadata else None,
                    record.timestamp.isoformat(),
                ),
            )
        self._conn.commit()
        logger.debug(f"Flushed {len(records)} token usage records")

    async def _flush_debate_buffer(self) -> None:
        """Flush debate usage buffer to database."""
        if not self._debate_buffer:
            return

        records = self._debate_buffer.copy()
        self._debate_buffer.clear()

        cursor = self._conn.cursor()
        for record in records:
            cursor.execute(
                """
                INSERT INTO debate_usage
                (id, org_id, user_id, debate_id, agent_count, rounds,
                 total_tokens, total_cost, duration_seconds, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.org_id,
                    record.user_id,
                    record.debate_id,
                    record.agent_count,
                    record.rounds,
                    record.total_tokens,
                    str(record.total_cost),
                    record.duration_seconds,
                    json.dumps(record.metadata) if record.metadata else None,
                    record.timestamp.isoformat(),
                ),
            )
        self._conn.commit()
        logger.debug(f"Flushed {len(records)} debate usage records")

    async def _flush_api_buffer(self) -> None:
        """Flush API call buffer to database."""
        if not self._api_buffer:
            return

        records = self._api_buffer.copy()
        self._api_buffer.clear()

        cursor = self._conn.cursor()
        for record in records:
            cursor.execute(
                """
                INSERT INTO api_usage
                (id, org_id, user_id, endpoint, method, status_code,
                 response_time_ms, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.org_id,
                    record.user_id,
                    record.endpoint,
                    record.method,
                    record.status_code,
                    record.response_time_ms,
                    json.dumps(record.metadata) if record.metadata else None,
                    record.timestamp.isoformat(),
                ),
            )
        self._conn.commit()
        logger.debug(f"Flushed {len(records)} API call records")

    async def flush_all(self) -> None:
        """Flush all buffers to database."""
        async with self._lock:
            await self._flush_token_buffer()
            await self._flush_debate_buffer()
            await self._flush_api_buffer()

    def _get_period_dates(
        self,
        period: str,
        reference_date: Optional[datetime] = None,
    ) -> Tuple[datetime, datetime]:
        """
        Get start and end dates for a period.

        Args:
            period: Period type (hour, day, week, month, quarter, year)
            reference_date: Reference date (default: now)

        Returns:
            Tuple of (start_date, end_date)
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)

        if period == "hour":
            start = reference_date.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif period == "day":
            start = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == "week":
            start = reference_date - timedelta(days=reference_date.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif period == "month":
            start = reference_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Get first day of next month
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        elif period == "quarter":
            quarter_month = ((reference_date.month - 1) // 3) * 3 + 1
            start = reference_date.replace(
                month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            if quarter_month + 3 > 12:
                end = start.replace(year=start.year + 1, month=(quarter_month + 3) % 12 or 12)
            else:
                end = start.replace(month=quarter_month + 3)
        elif period == "year":
            start = reference_date.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            end = start.replace(year=start.year + 1)
        else:
            # Default to month
            start = reference_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)

        return start, end

    async def get_usage_summary(
        self,
        org_id: str,
        period: str = "month",
        tier: str = "enterprise_plus",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> UsageSummary:
        """
        Get usage summary for a billing period.

        Args:
            org_id: Organization identifier
            period: Period type (hour, day, week, month, quarter, year)
            tier: Subscription tier for limits
            start_date: Optional explicit start date
            end_date: Optional explicit end date

        Returns:
            Usage summary with costs and limits
        """
        if not self._initialized:
            await self.initialize()

        # Get period dates
        if start_date is None or end_date is None:
            start_date, end_date = self._get_period_dates(period)

        summary = UsageSummary(
            org_id=org_id,
            period_start=start_date,
            period_end=end_date,
            period_type=(
                MeteringPeriod(period)
                if period in [p.value for p in MeteringPeriod]
                else MeteringPeriod.MONTH
            ),
        )

        cursor = self._conn.cursor()

        # Query hourly aggregates for the period
        cursor.execute(
            """
            SELECT
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(CAST(token_cost AS REAL)) as token_cost,
                SUM(debate_count) as debate_count,
                SUM(api_call_count) as api_call_count
            FROM hourly_aggregates
            WHERE org_id = ?
              AND hour >= ?
              AND hour < ?
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        row = cursor.fetchone()

        if row:
            summary.input_tokens = row["input_tokens"] or 0
            summary.output_tokens = row["output_tokens"] or 0
            summary.total_tokens = row["total_tokens"] or 0
            summary.token_cost = Decimal(str(row["token_cost"] or 0))
            summary.debate_count = row["debate_count"] or 0
            summary.api_call_count = row["api_call_count"] or 0

        # Get breakdown by model
        cursor.execute(
            """
            SELECT provider || '/' || model as model_key,
                   SUM(total_tokens) as tokens,
                   SUM(CAST(total_cost AS REAL)) as cost
            FROM token_usage
            WHERE org_id = ?
              AND timestamp >= ?
              AND timestamp < ?
            GROUP BY provider, model
            ORDER BY cost DESC
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        for row in cursor:
            summary.tokens_by_model[row["model_key"]] = row["tokens"] or 0
            summary.cost_by_model[row["model_key"]] = Decimal(str(row["cost"] or 0))

        # Get breakdown by provider
        cursor.execute(
            """
            SELECT provider,
                   SUM(total_tokens) as tokens,
                   SUM(CAST(total_cost AS REAL)) as cost
            FROM token_usage
            WHERE org_id = ?
              AND timestamp >= ?
              AND timestamp < ?
            GROUP BY provider
            ORDER BY cost DESC
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        for row in cursor:
            summary.tokens_by_provider[row["provider"]] = row["tokens"] or 0
            summary.cost_by_provider[row["provider"]] = Decimal(str(row["cost"] or 0))

        # Get breakdown by day
        cursor.execute(
            """
            SELECT DATE(hour) as day,
                   SUM(CAST(token_cost AS REAL)) as cost
            FROM hourly_aggregates
            WHERE org_id = ?
              AND hour >= ?
              AND hour < ?
            GROUP BY DATE(hour)
            ORDER BY day
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        for row in cursor:
            summary.cost_by_day[row["day"]] = Decimal(str(row["cost"] or 0))

        # Set limits based on tier
        caps = TIER_USAGE_CAPS.get(tier, TIER_USAGE_CAPS["enterprise_plus"])
        summary.token_limit = caps["max_tokens"]
        summary.debate_limit = caps["max_debates"]
        summary.api_call_limit = caps["max_api_calls"]

        # Calculate usage percentages
        if summary.token_limit > 0:
            summary.token_usage_percent = (summary.total_tokens / summary.token_limit) * 100
        if summary.debate_limit > 0:
            summary.debate_usage_percent = (summary.debate_count / summary.debate_limit) * 100
        if summary.api_call_limit > 0:
            summary.api_call_usage_percent = (summary.api_call_count / summary.api_call_limit) * 100

        return summary

    async def get_usage_breakdown(
        self,
        org_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> UsageBreakdown:
        """
        Get detailed usage breakdown for billing.

        Args:
            org_id: Organization identifier
            start_date: Start of period
            end_date: End of period

        Returns:
            Detailed usage breakdown
        """
        if not self._initialized:
            await self.initialize()

        if start_date is None or end_date is None:
            start_date, end_date = self._get_period_dates("month")

        breakdown = UsageBreakdown(
            org_id=org_id,
            period_start=start_date,
            period_end=end_date,
        )

        cursor = self._conn.cursor()

        # Get totals
        cursor.execute(
            """
            SELECT
                SUM(CAST(token_cost AS REAL)) as total_cost,
                SUM(total_tokens) as total_tokens
            FROM hourly_aggregates
            WHERE org_id = ?
              AND hour >= ?
              AND hour < ?
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        row = cursor.fetchone()
        if row:
            breakdown.total_cost = Decimal(str(row["total_cost"] or 0))
            breakdown.total_tokens = row["total_tokens"] or 0

        cursor.execute(
            """
            SELECT COUNT(*) as count FROM debate_usage
            WHERE org_id = ? AND timestamp >= ? AND timestamp < ?
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        row = cursor.fetchone()
        breakdown.total_debates = row["count"] if row else 0

        cursor.execute(
            """
            SELECT COUNT(*) as count FROM api_usage
            WHERE org_id = ? AND timestamp >= ? AND timestamp < ?
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        row = cursor.fetchone()
        breakdown.total_api_calls = row["count"] if row else 0

        # By model
        cursor.execute(
            """
            SELECT provider || '/' || model as model_key,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(total_tokens) as total_tokens,
                   SUM(CAST(total_cost AS REAL)) as cost,
                   COUNT(*) as requests
            FROM token_usage
            WHERE org_id = ? AND timestamp >= ? AND timestamp < ?
            GROUP BY provider, model
            ORDER BY cost DESC
            LIMIT 20
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        for row in cursor:
            breakdown.by_model.append(
                {
                    "model": row["model_key"],
                    "input_tokens": row["input_tokens"] or 0,
                    "output_tokens": row["output_tokens"] or 0,
                    "total_tokens": row["total_tokens"] or 0,
                    "cost": str(Decimal(str(row["cost"] or 0)).quantize(Decimal("0.0001"))),
                    "requests": row["requests"] or 0,
                }
            )

        # By provider
        cursor.execute(
            """
            SELECT provider,
                   SUM(total_tokens) as total_tokens,
                   SUM(CAST(total_cost AS REAL)) as cost,
                   COUNT(*) as requests
            FROM token_usage
            WHERE org_id = ? AND timestamp >= ? AND timestamp < ?
            GROUP BY provider
            ORDER BY cost DESC
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        for row in cursor:
            breakdown.by_provider.append(
                {
                    "provider": row["provider"],
                    "total_tokens": row["total_tokens"] or 0,
                    "cost": str(Decimal(str(row["cost"] or 0)).quantize(Decimal("0.0001"))),
                    "requests": row["requests"] or 0,
                }
            )

        # By day
        cursor.execute(
            """
            SELECT DATE(hour) as day,
                   SUM(total_tokens) as total_tokens,
                   SUM(CAST(token_cost AS REAL)) as cost,
                   SUM(debate_count) as debates,
                   SUM(api_call_count) as api_calls
            FROM hourly_aggregates
            WHERE org_id = ? AND hour >= ? AND hour < ?
            GROUP BY DATE(hour)
            ORDER BY day
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        for row in cursor:
            breakdown.by_day.append(
                {
                    "day": row["day"],
                    "total_tokens": row["total_tokens"] or 0,
                    "cost": str(Decimal(str(row["cost"] or 0)).quantize(Decimal("0.0001"))),
                    "debates": row["debates"] or 0,
                    "api_calls": row["api_calls"] or 0,
                }
            )

        # By user
        cursor.execute(
            """
            SELECT COALESCE(user_id, 'unknown') as user_id,
                   SUM(total_tokens) as total_tokens,
                   SUM(CAST(total_cost AS REAL)) as cost,
                   COUNT(*) as requests
            FROM token_usage
            WHERE org_id = ? AND timestamp >= ? AND timestamp < ?
            GROUP BY user_id
            ORDER BY cost DESC
            LIMIT 50
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        for row in cursor:
            breakdown.by_user.append(
                {
                    "user_id": row["user_id"],
                    "total_tokens": row["total_tokens"] or 0,
                    "cost": str(Decimal(str(row["cost"] or 0)).quantize(Decimal("0.0001"))),
                    "requests": row["requests"] or 0,
                }
            )

        return breakdown

    async def get_usage_limits(
        self,
        org_id: str,
        tier: str = "enterprise_plus",
    ) -> UsageLimits:
        """
        Get current usage limits and utilization.

        Args:
            org_id: Organization identifier
            tier: Subscription tier

        Returns:
            Current limits and usage percentages
        """
        if not self._initialized:
            await self.initialize()

        # Get current month's usage
        start_date, end_date = self._get_period_dates("month")

        limits = UsageLimits(
            org_id=org_id,
            tier=tier,
        )

        # Set limits based on tier
        caps = TIER_USAGE_CAPS.get(tier, TIER_USAGE_CAPS["enterprise_plus"])
        limits.max_tokens = caps["max_tokens"]
        limits.max_debates = caps["max_debates"]
        limits.max_api_calls = caps["max_api_calls"]

        # Get current usage from aggregates
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                SUM(total_tokens) as tokens,
                SUM(debate_count) as debates,
                SUM(api_call_count) as api_calls
            FROM hourly_aggregates
            WHERE org_id = ?
              AND hour >= ?
              AND hour < ?
            """,
            (org_id, start_date.isoformat(), end_date.isoformat()),
        )
        row = cursor.fetchone()

        if row:
            limits.tokens_used = row["tokens"] or 0
            limits.debates_used = row["debates"] or 0
            limits.api_calls_used = row["api_calls"] or 0

        # Calculate percentages
        if limits.max_tokens > 0:
            limits.tokens_percent = (limits.tokens_used / limits.max_tokens) * 100
            limits.tokens_exceeded = limits.tokens_used >= limits.max_tokens
        if limits.max_debates > 0:
            limits.debates_percent = (limits.debates_used / limits.max_debates) * 100
            limits.debates_exceeded = limits.debates_used >= limits.max_debates
        if limits.max_api_calls > 0:
            limits.api_calls_percent = (limits.api_calls_used / limits.max_api_calls) * 100
            limits.api_calls_exceeded = limits.api_calls_used >= limits.max_api_calls

        return limits

    async def close(self) -> None:
        """Close resources and flush buffers."""
        await self.flush_all()
        if self._conn:
            self._conn.close()
            self._conn = None


# Module-level instance
_usage_meter: Optional[UsageMeter] = None


def get_usage_meter() -> UsageMeter:
    """Get or create the global usage meter."""
    global _usage_meter
    if _usage_meter is None:
        _usage_meter = UsageMeter()
    return _usage_meter


__all__ = [
    "UsageMeter",
    "UsageSummary",
    "UsageBreakdown",
    "UsageLimits",
    "TokenUsageRecord",
    "DebateUsageRecord",
    "ApiCallRecord",
    "HourlyAggregate",
    "MeteringPeriod",
    "UsageType",
    "MODEL_PRICING",
    "TIER_USAGE_CAPS",
    "get_usage_meter",
]
