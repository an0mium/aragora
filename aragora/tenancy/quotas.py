"""
Tenant quota and rate limiting management.

Enforces resource limits per tenant including API rate limits,
storage quotas, and usage caps.

Usage:
    from aragora.tenancy import QuotaManager, QuotaConfig, QuotaExceeded

    manager = QuotaManager(config)

    # Check before operation
    if not await manager.check_quota("debates", 1):
        raise QuotaExceeded("Debate limit reached")

    # Or use with automatic checking
    await manager.consume("api_requests", 1)

    # Get usage stats
    stats = await manager.get_usage_stats()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from aragora.tenancy.context import get_current_tenant, get_current_tenant_id

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class QuotaPeriod(Enum):
    """Time period for quota limits."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    UNLIMITED = "unlimited"


class QuotaExceeded(Exception):
    """Raised when a quota limit is exceeded."""

    def __init__(
        self,
        message: str,
        resource: str,
        limit: int,
        current: int,
        period: QuotaPeriod,
        tenant_id: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        self.resource = resource
        self.limit = limit
        self.current = current
        self.period = period
        self.tenant_id = tenant_id
        self.retry_after = retry_after
        super().__init__(message)


@dataclass
class QuotaLimit:
    """A single quota limit configuration."""

    resource: str
    """Resource being limited (e.g., 'api_requests', 'debates', 'storage_bytes')."""

    limit: int
    """Maximum allowed value."""

    period: QuotaPeriod = QuotaPeriod.DAY
    """Time period for the limit."""

    soft_limit: Optional[int] = None
    """Warning threshold (optional)."""

    burst_limit: Optional[int] = None
    """Short-term burst allowance (optional)."""

    @property
    def period_seconds(self) -> int:
        """Get period duration in seconds."""
        return {
            QuotaPeriod.MINUTE: 60,
            QuotaPeriod.HOUR: 3600,
            QuotaPeriod.DAY: 86400,
            QuotaPeriod.WEEK: 604800,
            QuotaPeriod.MONTH: 2592000,  # 30 days
            QuotaPeriod.UNLIMITED: 0,
        }.get(self.period, 86400)


@dataclass
class QuotaConfig:
    """Configuration for quota management."""

    # Default limits
    limits: list[QuotaLimit] = field(default_factory=list)
    """List of quota limits to enforce."""

    # Behavior
    strict_enforcement: bool = True
    """Raise exceptions on quota exceeded."""

    warn_at_threshold: float = 0.8
    """Warn when usage reaches this percentage."""

    # Rate limiting
    enable_rate_limiting: bool = True
    """Enable rate limiting."""

    rate_limit_window: int = 60
    """Rate limit window in seconds."""

    # Persistence
    persist_usage: bool = True
    """Persist usage data."""

    cleanup_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    """How often to clean up old usage data."""

    @classmethod
    def default_limits(cls) -> "QuotaConfig":
        """Get default quota configuration."""
        return cls(
            limits=[
                # API rate limits
                QuotaLimit("api_requests", 60, QuotaPeriod.MINUTE, burst_limit=100),
                QuotaLimit("api_requests", 10000, QuotaPeriod.DAY),

                # Debate limits
                QuotaLimit("debates", 100, QuotaPeriod.DAY),
                QuotaLimit("concurrent_debates", 5, QuotaPeriod.UNLIMITED),

                # Token limits
                QuotaLimit("tokens", 1_000_000, QuotaPeriod.MONTH),
                QuotaLimit("tokens_per_debate", 50_000, QuotaPeriod.UNLIMITED),

                # Storage limits
                QuotaLimit("storage_bytes", 10 * 1024 * 1024 * 1024, QuotaPeriod.UNLIMITED),  # 10GB
                QuotaLimit("knowledge_bytes", 1 * 1024 * 1024 * 1024, QuotaPeriod.UNLIMITED),  # 1GB

                # User limits
                QuotaLimit("users", 10, QuotaPeriod.UNLIMITED),
                QuotaLimit("connectors", 5, QuotaPeriod.UNLIMITED),
            ]
        )


@dataclass
class UsageRecord:
    """Record of resource usage."""

    resource: str
    tenant_id: str
    count: int
    period_start: datetime
    period: QuotaPeriod
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class QuotaStatus:
    """Status of a quota for a tenant."""

    resource: str
    limit: int
    current: int
    remaining: int
    period: QuotaPeriod
    period_resets_at: Optional[datetime]
    percentage_used: float
    is_exceeded: bool
    is_warning: bool


class QuotaManager:
    """
    Manages tenant quotas and rate limits.

    Tracks usage across multiple resources and enforces limits
    based on tenant configuration.
    """

    def __init__(self, config: Optional[QuotaConfig] = None):
        """Initialize quota manager."""
        self.config = config or QuotaConfig.default_limits()
        self._usage: dict[str, dict[str, UsageRecord]] = defaultdict(dict)
        self._rate_limiters: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._lock = asyncio.Lock()
        self._limits_cache: dict[str, dict[str, QuotaLimit]] = {}

    def _get_limits_for_tenant(self, tenant_id: str) -> dict[str, QuotaLimit]:
        """Get quota limits for a tenant."""
        if tenant_id in self._limits_cache:
            return self._limits_cache[tenant_id]

        # Start with default limits
        limits = {limit.resource: limit for limit in self.config.limits}

        # Override with tenant-specific limits if tenant is available
        tenant = get_current_tenant()
        if tenant and tenant.config:
            # Map tenant config to quota limits
            config = tenant.config
            overrides = {
                "api_requests": QuotaLimit(
                    "api_requests",
                    config.api_requests_per_minute,
                    QuotaPeriod.MINUTE,
                ),
                "debates": QuotaLimit(
                    "debates",
                    config.max_debates_per_day,
                    QuotaPeriod.DAY,
                ),
                "concurrent_debates": QuotaLimit(
                    "concurrent_debates",
                    config.max_concurrent_debates,
                    QuotaPeriod.UNLIMITED,
                ),
                "tokens": QuotaLimit(
                    "tokens",
                    config.tokens_per_month,
                    QuotaPeriod.MONTH,
                ),
                "tokens_per_debate": QuotaLimit(
                    "tokens_per_debate",
                    config.tokens_per_debate,
                    QuotaPeriod.UNLIMITED,
                ),
                "storage_bytes": QuotaLimit(
                    "storage_bytes",
                    config.storage_quota,
                    QuotaPeriod.UNLIMITED,
                ),
                "knowledge_bytes": QuotaLimit(
                    "knowledge_bytes",
                    config.knowledge_quota,
                    QuotaPeriod.UNLIMITED,
                ),
                "users": QuotaLimit(
                    "users",
                    config.max_users,
                    QuotaPeriod.UNLIMITED,
                ),
                "connectors": QuotaLimit(
                    "connectors",
                    config.max_connectors,
                    QuotaPeriod.UNLIMITED,
                ),
            }
            limits.update(overrides)

        self._limits_cache[tenant_id] = limits
        return limits

    def _get_period_key(self, period: QuotaPeriod) -> str:
        """Get a key for the current period."""
        now = datetime.now()
        if period == QuotaPeriod.MINUTE:
            return now.strftime("%Y-%m-%d-%H-%M")
        if period == QuotaPeriod.HOUR:
            return now.strftime("%Y-%m-%d-%H")
        if period == QuotaPeriod.DAY:
            return now.strftime("%Y-%m-%d")
        if period == QuotaPeriod.WEEK:
            return f"{now.year}-W{now.isocalendar()[1]}"
        if period == QuotaPeriod.MONTH:
            return now.strftime("%Y-%m")
        return "unlimited"

    def _get_period_start(self, period: QuotaPeriod) -> datetime:
        """Get the start of the current period."""
        now = datetime.now()
        if period == QuotaPeriod.MINUTE:
            return now.replace(second=0, microsecond=0)
        if period == QuotaPeriod.HOUR:
            return now.replace(minute=0, second=0, microsecond=0)
        if period == QuotaPeriod.DAY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        if period == QuotaPeriod.WEEK:
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if period == QuotaPeriod.MONTH:
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return datetime.min

    def _get_period_end(self, period: QuotaPeriod) -> Optional[datetime]:
        """Get the end of the current period."""
        start = self._get_period_start(period)
        if period == QuotaPeriod.MINUTE:
            return start + timedelta(minutes=1)
        if period == QuotaPeriod.HOUR:
            return start + timedelta(hours=1)
        if period == QuotaPeriod.DAY:
            return start + timedelta(days=1)
        if period == QuotaPeriod.WEEK:
            return start + timedelta(weeks=1)
        if period == QuotaPeriod.MONTH:
            # Approximate - next month
            if start.month == 12:
                return start.replace(year=start.year + 1, month=1)
            return start.replace(month=start.month + 1)
        return None

    async def check_quota(
        self,
        resource: str,
        amount: int = 1,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """
        Check if quota allows the requested amount.

        Args:
            resource: Resource to check
            amount: Amount to consume
            tenant_id: Tenant ID (uses current context if not provided)

        Returns:
            True if quota allows, False otherwise
        """
        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            return True  # No tenant = no limits

        limits = self._get_limits_for_tenant(tid)
        limit_config = limits.get(resource)
        if limit_config is None:
            return True  # No limit configured

        if limit_config.period == QuotaPeriod.UNLIMITED:
            # For unlimited period, check absolute value
            current = await self._get_current_usage(tid, resource, limit_config.period)
            return current + amount <= limit_config.limit

        # Check period-based limit
        current = await self._get_current_usage(tid, resource, limit_config.period)
        return current + amount <= limit_config.limit

    async def consume(
        self,
        resource: str,
        amount: int = 1,
        tenant_id: Optional[str] = None,
    ) -> None:
        """
        Consume quota for a resource.

        Args:
            resource: Resource to consume
            amount: Amount to consume
            tenant_id: Tenant ID (uses current context if not provided)

        Raises:
            QuotaExceeded: If quota would be exceeded
        """
        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            return  # No tenant = no tracking

        limits = self._get_limits_for_tenant(tid)
        limit_config = limits.get(resource)

        if limit_config is not None and self.config.strict_enforcement:
            current = await self._get_current_usage(tid, resource, limit_config.period)
            if current + amount > limit_config.limit:
                period_end = self._get_period_end(limit_config.period)
                retry_after = None
                if period_end:
                    retry_after = int((period_end - datetime.now()).total_seconds())

                raise QuotaExceeded(
                    f"Quota exceeded for {resource}: {current + amount} > {limit_config.limit}",
                    resource=resource,
                    limit=limit_config.limit,
                    current=current,
                    period=limit_config.period,
                    tenant_id=tid,
                    retry_after=retry_after,
                )

        await self._record_usage(tid, resource, amount)

    async def _get_current_usage(
        self,
        tenant_id: str,
        resource: str,
        period: QuotaPeriod,
    ) -> int:
        """Get current usage for a resource."""
        key = f"{resource}:{self._get_period_key(period)}"

        async with self._lock:
            if tenant_id in self._usage and key in self._usage[tenant_id]:
                record = self._usage[tenant_id][key]
                return record.count
            return 0

    async def _record_usage(
        self,
        tenant_id: str,
        resource: str,
        amount: int,
    ) -> None:
        """Record usage of a resource."""
        limits = self._get_limits_for_tenant(tenant_id)
        limit_config = limits.get(resource)
        period = limit_config.period if limit_config else QuotaPeriod.DAY

        key = f"{resource}:{self._get_period_key(period)}"

        async with self._lock:
            if key not in self._usage[tenant_id]:
                self._usage[tenant_id][key] = UsageRecord(
                    resource=resource,
                    tenant_id=tenant_id,
                    count=0,
                    period_start=self._get_period_start(period),
                    period=period,
                )

            self._usage[tenant_id][key].count += amount
            self._usage[tenant_id][key].last_updated = datetime.now()

    async def check_rate_limit(
        self,
        resource: str,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """
        Check rate limit for a resource.

        Args:
            resource: Resource to check
            tenant_id: Tenant ID (uses current context if not provided)

        Returns:
            True if within rate limit
        """
        if not self.config.enable_rate_limiting:
            return True

        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            return True

        limits = self._get_limits_for_tenant(tid)
        limit_config = limits.get(resource)
        if limit_config is None or limit_config.period != QuotaPeriod.MINUTE:
            return True

        now = time.time()
        window = self.config.rate_limit_window

        async with self._lock:
            timestamps = self._rate_limiters[tid][resource]

            # Remove old timestamps
            timestamps[:] = [ts for ts in timestamps if now - ts < window]

            # Check limit (use burst_limit if available)
            max_requests = limit_config.burst_limit or limit_config.limit
            return len(timestamps) < max_requests

    async def record_rate_limit(
        self,
        resource: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Record a request for rate limiting."""
        if not self.config.enable_rate_limiting:
            return

        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            return

        async with self._lock:
            self._rate_limiters[tid][resource].append(time.time())

    async def get_quota_status(
        self,
        resource: str,
        tenant_id: Optional[str] = None,
    ) -> QuotaStatus:
        """
        Get status of a quota.

        Args:
            resource: Resource to check
            tenant_id: Tenant ID (uses current context if not provided)

        Returns:
            QuotaStatus with current state
        """
        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            raise ValueError("No tenant context")

        limits = self._get_limits_for_tenant(tid)
        limit_config = limits.get(resource)
        if limit_config is None:
            raise ValueError(f"No quota configured for {resource}")

        current = await self._get_current_usage(tid, resource, limit_config.period)
        remaining = max(0, limit_config.limit - current)
        percentage = (current / limit_config.limit * 100) if limit_config.limit > 0 else 0

        return QuotaStatus(
            resource=resource,
            limit=limit_config.limit,
            current=current,
            remaining=remaining,
            period=limit_config.period,
            period_resets_at=self._get_period_end(limit_config.period),
            percentage_used=percentage,
            is_exceeded=current >= limit_config.limit,
            is_warning=percentage >= self.config.warn_at_threshold * 100,
        )

    async def get_all_quotas(
        self,
        tenant_id: Optional[str] = None,
    ) -> list[QuotaStatus]:
        """Get status of all quotas for a tenant."""
        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            raise ValueError("No tenant context")

        limits = self._get_limits_for_tenant(tid)
        statuses = []

        for resource in limits:
            status = await self.get_quota_status(resource, tid)
            statuses.append(status)

        return statuses

    async def get_usage_stats(
        self,
        tenant_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get usage statistics for a tenant.

        Returns:
            Dictionary with usage stats by resource
        """
        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            raise ValueError("No tenant context")

        quotas = await self.get_all_quotas(tid)

        return {
            "tenant_id": tid,
            "timestamp": datetime.now().isoformat(),
            "quotas": [
                {
                    "resource": q.resource,
                    "limit": q.limit,
                    "current": q.current,
                    "remaining": q.remaining,
                    "percentage_used": round(q.percentage_used, 2),
                    "period": q.period.value,
                    "resets_at": q.period_resets_at.isoformat() if q.period_resets_at else None,
                    "status": "exceeded" if q.is_exceeded else "warning" if q.is_warning else "ok",
                }
                for q in quotas
            ],
        }

    async def reset_usage(
        self,
        resource: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Reset usage for a resource."""
        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            return

        async with self._lock:
            keys_to_remove = [
                key for key in self._usage.get(tid, {})
                if key.startswith(f"{resource}:")
            ]
            for key in keys_to_remove:
                del self._usage[tid][key]

    async def reset_all_usage(self, tenant_id: Optional[str] = None) -> None:
        """Reset all usage for a tenant."""
        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            return

        async with self._lock:
            if tid in self._usage:
                self._usage[tid].clear()
            if tid in self._rate_limiters:
                self._rate_limiters[tid].clear()

    def invalidate_limits_cache(self, tenant_id: Optional[str] = None) -> None:
        """Invalidate limits cache when tenant config changes."""
        if tenant_id:
            self._limits_cache.pop(tenant_id, None)
        else:
            self._limits_cache.clear()

    async def cleanup_old_usage(self) -> int:
        """
        Clean up old usage records.

        Returns:
            Number of records cleaned up
        """
        cleaned = 0
        now = datetime.now()

        async with self._lock:
            for tenant_id in list(self._usage.keys()):
                keys_to_remove = []
                for key, record in self._usage[tenant_id].items():
                    period_end = self._get_period_end(record.period)
                    if period_end and period_end < now:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self._usage[tenant_id][key]
                    cleaned += 1

        logger.debug(f"Cleaned up {cleaned} old usage records")
        return cleaned
