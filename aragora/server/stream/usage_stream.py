"""
Real-time Usage Metering Stream.

Provides WebSocket streaming for real-time usage metrics:
- Token consumption events
- Cost updates
- Budget alerts
- Usage trends

Usage:
    from aragora.server.stream.usage_stream import (
        UsageStreamEmitter,
        get_usage_emitter,
        emit_usage_event,
    )

    # Emit usage event
    await emit_usage_event(
        tenant_id="tenant-123",
        event_type=UsageEventType.TOKEN_USAGE,
        data={"tokens": 1000, "cost": 0.05},
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class UsageEventType(Enum):
    """Types of usage stream events."""

    TOKEN_USAGE = "token_usage"  # Individual token consumption
    COST_UPDATE = "cost_update"  # Cost accumulation update
    BUDGET_ALERT = "budget_alert"  # Budget threshold reached
    USAGE_SUMMARY = "usage_summary"  # Periodic summary
    RATE_LIMIT = "rate_limit"  # Rate limit event
    QUOTA_WARNING = "quota_warning"  # Approaching quota


@dataclass
class UsageStreamEvent:
    """A real-time usage event."""

    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: UsageEventType = UsageEventType.TOKEN_USAGE
    tenant_id: str = ""
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None

    # Usage data
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    cost_usd: Decimal = Decimal("0")

    # Context
    provider: str = ""
    model: str = ""
    debate_id: Optional[str] = None
    operation: str = ""

    # Budget tracking
    budget_used_pct: float = 0.0
    budget_remaining: Decimal = Decimal("0")

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "cost_usd": str(self.cost_usd),
            "provider": self.provider,
            "model": self.model,
            "debate_id": self.debate_id,
            "operation": self.operation,
            "budget_used_pct": self.budget_used_pct,
            "budget_remaining": str(self.budget_remaining),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class UsageStreamEmitter:
    """
    Real-time usage stream emitter.

    Broadcasts usage events to connected WebSocket clients.
    Supports:
    - Tenant-scoped subscriptions
    - Workspace-scoped subscriptions
    - Aggregated usage summaries
    - Budget alert thresholds
    """

    def __init__(self):
        """Initialize usage stream emitter."""
        self._subscribers: Dict[str, Set[Callable]] = {}  # tenant_id -> callbacks
        self._workspace_subscribers: Dict[str, Set[Callable]] = {}
        self._global_subscribers: Set[Callable] = set()
        self._lock = asyncio.Lock()

        # Aggregation state
        self._current_period_usage: Dict[str, Dict[str, Any]] = {}
        self._summary_interval_seconds = 60  # Emit summaries every minute

        logger.info("[UsageStream] Emitter initialized")

    async def subscribe_tenant(
        self,
        tenant_id: str,
        callback: Callable[[UsageStreamEvent], None],
    ) -> None:
        """Subscribe to usage events for a tenant."""
        async with self._lock:
            if tenant_id not in self._subscribers:
                self._subscribers[tenant_id] = set()
            self._subscribers[tenant_id].add(callback)
            logger.debug(f"[UsageStream] Subscribed to tenant {tenant_id}")

    async def unsubscribe_tenant(
        self,
        tenant_id: str,
        callback: Callable[[UsageStreamEvent], None],
    ) -> None:
        """Unsubscribe from tenant events."""
        async with self._lock:
            if tenant_id in self._subscribers:
                self._subscribers[tenant_id].discard(callback)
                if not self._subscribers[tenant_id]:
                    del self._subscribers[tenant_id]

    async def subscribe_workspace(
        self,
        workspace_id: str,
        callback: Callable[[UsageStreamEvent], None],
    ) -> None:
        """Subscribe to usage events for a workspace."""
        async with self._lock:
            if workspace_id not in self._workspace_subscribers:
                self._workspace_subscribers[workspace_id] = set()
            self._workspace_subscribers[workspace_id].add(callback)

    async def subscribe_global(
        self,
        callback: Callable[[UsageStreamEvent], None],
    ) -> None:
        """Subscribe to all usage events (admin only)."""
        async with self._lock:
            self._global_subscribers.add(callback)

    async def emit(self, event: UsageStreamEvent) -> None:
        """
        Emit a usage event to all relevant subscribers.

        Args:
            event: Usage event to emit
        """
        async with self._lock:
            callbacks = set()

            # Global subscribers (admin dashboards)
            callbacks.update(self._global_subscribers)

            # Tenant subscribers
            if event.tenant_id and event.tenant_id in self._subscribers:
                callbacks.update(self._subscribers[event.tenant_id])

            # Workspace subscribers
            if event.workspace_id and event.workspace_id in self._workspace_subscribers:
                callbacks.update(self._workspace_subscribers[event.workspace_id])

        # Emit to all callbacks
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.warning(f"[UsageStream] Callback error: {e}")

        # Update aggregation state
        self._update_aggregation(event)

        logger.debug(
            f"[UsageStream] Emitted {event.event_type.value} to {len(callbacks)} subscribers"
        )

    def _update_aggregation(self, event: UsageStreamEvent) -> None:
        """Update aggregation state for periodic summaries."""
        key = event.tenant_id or "global"
        if key not in self._current_period_usage:
            self._current_period_usage[key] = {
                "tokens_in": 0,
                "tokens_out": 0,
                "total_cost": Decimal("0"),
                "event_count": 0,
            }

        agg = self._current_period_usage[key]
        agg["tokens_in"] += event.tokens_in
        agg["tokens_out"] += event.tokens_out
        agg["total_cost"] += event.cost_usd
        agg["event_count"] += 1

    async def emit_summary(self, tenant_id: str) -> UsageStreamEvent:
        """
        Emit a usage summary event.

        Args:
            tenant_id: Tenant to summarize

        Returns:
            Summary event
        """
        key = tenant_id or "global"
        agg = self._current_period_usage.get(key, {})

        event = UsageStreamEvent(
            event_type=UsageEventType.USAGE_SUMMARY,
            tenant_id=tenant_id,
            tokens_in=agg.get("tokens_in", 0),
            tokens_out=agg.get("tokens_out", 0),
            total_tokens=agg.get("tokens_in", 0) + agg.get("tokens_out", 0),
            cost_usd=agg.get("total_cost", Decimal("0")),
            metadata={"event_count": agg.get("event_count", 0)},
        )

        await self.emit(event)

        # Reset aggregation
        if key in self._current_period_usage:
            del self._current_period_usage[key]

        return event

    async def emit_budget_alert(
        self,
        tenant_id: str,
        budget_used_pct: float,
        budget_remaining: Decimal,
        alert_level: str = "warning",
    ) -> None:
        """
        Emit a budget alert event.

        Args:
            tenant_id: Tenant ID
            budget_used_pct: Percentage of budget used
            budget_remaining: Remaining budget amount
            alert_level: Alert severity (info, warning, critical, exceeded)
        """
        event = UsageStreamEvent(
            event_type=UsageEventType.BUDGET_ALERT,
            tenant_id=tenant_id,
            budget_used_pct=budget_used_pct,
            budget_remaining=budget_remaining,
            metadata={"alert_level": alert_level},
        )
        await self.emit(event)

    def get_subscriber_count(self, tenant_id: Optional[str] = None) -> int:
        """Get count of subscribers."""
        if tenant_id:
            return len(self._subscribers.get(tenant_id, set()))
        return (
            len(self._global_subscribers)
            + sum(len(s) for s in self._subscribers.values())
            + sum(len(s) for s in self._workspace_subscribers.values())
        )


# Global emitter instance
_usage_emitter: Optional[UsageStreamEmitter] = None


def get_usage_emitter() -> UsageStreamEmitter:
    """Get or create global usage stream emitter."""
    global _usage_emitter
    if _usage_emitter is None:
        _usage_emitter = UsageStreamEmitter()
    return _usage_emitter


async def emit_usage_event(
    tenant_id: str,
    event_type: UsageEventType = UsageEventType.TOKEN_USAGE,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: Decimal = Decimal("0"),
    provider: str = "",
    model: str = "",
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    debate_id: Optional[str] = None,
    operation: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> UsageStreamEvent:
    """
    Convenience function to emit a usage event.

    Args:
        tenant_id: Tenant ID
        event_type: Type of event
        tokens_in: Input tokens
        tokens_out: Output tokens
        cost_usd: Cost in USD
        provider: Provider name
        model: Model name
        workspace_id: Workspace ID
        user_id: User ID
        debate_id: Debate ID
        operation: Operation name
        metadata: Additional metadata

    Returns:
        Emitted event
    """
    emitter = get_usage_emitter()

    event = UsageStreamEvent(
        event_type=event_type,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        user_id=user_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=tokens_in + tokens_out,
        cost_usd=cost_usd,
        provider=provider,
        model=model,
        debate_id=debate_id,
        operation=operation,
        metadata=metadata or {},
    )

    await emitter.emit(event)
    return event


__all__ = [
    "UsageEventType",
    "UsageStreamEvent",
    "UsageStreamEmitter",
    "get_usage_emitter",
    "emit_usage_event",
]
