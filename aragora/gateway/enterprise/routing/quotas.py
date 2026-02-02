"""
Quota Management for Multi-Tenant Router.

Provides sliding window rate limiting and quota tracking per tenant.

Features:
- Per-minute, per-hour, per-day rate limits
- Concurrent request limiting
- Bandwidth limiting
- Warning thresholds
- Thread-safe tracking with async locks

Usage:
    from aragora.gateway.enterprise.routing.quotas import (
        QuotaTracker,
        TenantQuotas,
        QuotaStatus,
    )

    tracker = QuotaTracker()
    quotas = TenantQuotas(requests_per_minute=100)

    allowed, status = await tracker.check_and_consume("tenant-1", quotas)
    if not allowed:
        print(f"Quota exceeded: {status.quota_type}")
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TenantQuotas:
    """
    Quota configuration for a tenant.

    Attributes:
        requests_per_minute: Maximum requests per minute.
        requests_per_hour: Maximum requests per hour.
        requests_per_day: Maximum requests per day.
        concurrent_requests: Maximum concurrent requests.
        bandwidth_bytes_per_minute: Maximum bandwidth in bytes per minute.
        warn_threshold: Percentage at which to emit warning (0.0-1.0).
    """

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    concurrent_requests: int = 10
    bandwidth_bytes_per_minute: int = 10 * 1024 * 1024  # 10 MB
    warn_threshold: float = 0.8


@dataclass
class QuotaStatus:
    """
    Current quota status for a tenant.

    Attributes:
        tenant_id: Tenant identifier.
        used: Current usage count/amount.
        remaining: Remaining quota.
        limit: Maximum quota limit.
        reset_time: When the quota resets.
        quota_type: Type of quota (requests_per_minute, etc.).
        is_exceeded: Whether quota is exceeded.
        is_warning: Whether quota is at warning threshold.
        percentage_used: Percentage of quota used (0-100).
    """

    tenant_id: str
    used: int
    remaining: int
    limit: int
    reset_time: datetime
    quota_type: str = "requests_per_minute"
    is_exceeded: bool = False
    is_warning: bool = False
    percentage_used: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tenant_id": self.tenant_id,
            "used": self.used,
            "remaining": self.remaining,
            "limit": self.limit,
            "reset_time": self.reset_time.isoformat(),
            "quota_type": self.quota_type,
            "is_exceeded": self.is_exceeded,
            "is_warning": self.is_warning,
            "percentage_used": round(self.percentage_used, 2),
        }


# =============================================================================
# Quota Tracker
# =============================================================================


class QuotaTracker:
    """
    Tracks quota usage per tenant with sliding window rate limiting.

    Thread-safe quota tracking with automatic window expiration.
    """

    def __init__(self) -> None:
        """Initialize quota tracker."""
        self._usage: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._concurrent: dict[str, int] = defaultdict(int)
        self._bandwidth: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = asyncio.Lock()

    async def check_and_consume(
        self,
        tenant_id: str,
        quotas: TenantQuotas,
        bytes_size: int = 0,
    ) -> tuple[bool, QuotaStatus | None]:
        """
        Check if request is within quota and record usage.

        Args:
            tenant_id: Tenant identifier.
            quotas: Quota configuration for the tenant.
            bytes_size: Size of request in bytes (for bandwidth limiting).

        Returns:
            Tuple of (is_allowed, exceeded_quota_status).
        """
        async with self._lock:
            now = time.time()

            # Check concurrent requests
            if self._concurrent[tenant_id] >= quotas.concurrent_requests:
                return False, QuotaStatus(
                    tenant_id=tenant_id,
                    used=self._concurrent[tenant_id],
                    remaining=0,
                    limit=quotas.concurrent_requests,
                    reset_time=datetime.utcnow(),
                    quota_type="concurrent_requests",
                    is_exceeded=True,
                    percentage_used=100.0,
                )

            # Check per-minute rate limit
            minute_timestamps = self._usage[tenant_id]["minute"]
            minute_cutoff = now - 60
            minute_timestamps[:] = [ts for ts in minute_timestamps if ts > minute_cutoff]

            if len(minute_timestamps) >= quotas.requests_per_minute:
                oldest = min(minute_timestamps) if minute_timestamps else now
                reset_time = datetime.utcfromtimestamp(oldest + 60)
                return False, QuotaStatus(
                    tenant_id=tenant_id,
                    used=len(minute_timestamps),
                    remaining=0,
                    limit=quotas.requests_per_minute,
                    reset_time=reset_time,
                    quota_type="requests_per_minute",
                    is_exceeded=True,
                    percentage_used=100.0,
                )

            # Check per-hour rate limit
            hour_timestamps = self._usage[tenant_id]["hour"]
            hour_cutoff = now - 3600
            hour_timestamps[:] = [ts for ts in hour_timestamps if ts > hour_cutoff]

            if len(hour_timestamps) >= quotas.requests_per_hour:
                oldest = min(hour_timestamps) if hour_timestamps else now
                reset_time = datetime.utcfromtimestamp(oldest + 3600)
                return False, QuotaStatus(
                    tenant_id=tenant_id,
                    used=len(hour_timestamps),
                    remaining=0,
                    limit=quotas.requests_per_hour,
                    reset_time=reset_time,
                    quota_type="requests_per_hour",
                    is_exceeded=True,
                    percentage_used=100.0,
                )

            # Check per-day rate limit
            day_timestamps = self._usage[tenant_id]["day"]
            day_cutoff = now - 86400
            day_timestamps[:] = [ts for ts in day_timestamps if ts > day_cutoff]

            if len(day_timestamps) >= quotas.requests_per_day:
                oldest = min(day_timestamps) if day_timestamps else now
                reset_time = datetime.utcfromtimestamp(oldest + 86400)
                return False, QuotaStatus(
                    tenant_id=tenant_id,
                    used=len(day_timestamps),
                    remaining=0,
                    limit=quotas.requests_per_day,
                    reset_time=reset_time,
                    quota_type="requests_per_day",
                    is_exceeded=True,
                    percentage_used=100.0,
                )

            # Check bandwidth limit
            if bytes_size > 0:
                minute_key = str(int(now // 60))
                current_bandwidth = self._bandwidth[tenant_id][minute_key]
                if current_bandwidth + bytes_size > quotas.bandwidth_bytes_per_minute:
                    reset_time = datetime.utcfromtimestamp((int(now // 60) + 1) * 60)
                    return False, QuotaStatus(
                        tenant_id=tenant_id,
                        used=current_bandwidth,
                        remaining=max(0, quotas.bandwidth_bytes_per_minute - current_bandwidth),
                        limit=quotas.bandwidth_bytes_per_minute,
                        reset_time=reset_time,
                        quota_type="bandwidth_bytes_per_minute",
                        is_exceeded=True,
                        percentage_used=(current_bandwidth / quotas.bandwidth_bytes_per_minute)
                        * 100,
                    )
                self._bandwidth[tenant_id][minute_key] += bytes_size

            # Record usage
            minute_timestamps.append(now)
            hour_timestamps.append(now)
            day_timestamps.append(now)
            self._concurrent[tenant_id] += 1

            return True, None

    async def release_concurrent(self, tenant_id: str) -> None:
        """
        Release a concurrent request slot.

        Args:
            tenant_id: Tenant identifier.
        """
        async with self._lock:
            self._concurrent[tenant_id] = max(0, self._concurrent[tenant_id] - 1)

    async def get_status(
        self,
        tenant_id: str,
        quotas: TenantQuotas,
    ) -> dict[str, QuotaStatus]:
        """
        Get current quota status for a tenant.

        Args:
            tenant_id: Tenant identifier.
            quotas: Quota configuration for the tenant.

        Returns:
            Dictionary of quota type to QuotaStatus.
        """
        async with self._lock:
            now = time.time()
            statuses: dict[str, QuotaStatus] = {}

            # Per-minute status
            minute_timestamps = self._usage[tenant_id]["minute"]
            minute_cutoff = now - 60
            minute_timestamps[:] = [ts for ts in minute_timestamps if ts > minute_cutoff]
            minute_used = len(minute_timestamps)
            minute_remaining = max(0, quotas.requests_per_minute - minute_used)
            minute_pct = (
                (minute_used / quotas.requests_per_minute * 100)
                if quotas.requests_per_minute > 0
                else 0
            )

            statuses["requests_per_minute"] = QuotaStatus(
                tenant_id=tenant_id,
                used=minute_used,
                remaining=minute_remaining,
                limit=quotas.requests_per_minute,
                reset_time=datetime.utcfromtimestamp(now + 60),
                quota_type="requests_per_minute",
                is_exceeded=minute_remaining == 0,
                is_warning=minute_pct >= quotas.warn_threshold * 100,
                percentage_used=minute_pct,
            )

            # Per-hour status
            hour_timestamps = self._usage[tenant_id]["hour"]
            hour_cutoff = now - 3600
            hour_timestamps[:] = [ts for ts in hour_timestamps if ts > hour_cutoff]
            hour_used = len(hour_timestamps)
            hour_remaining = max(0, quotas.requests_per_hour - hour_used)
            hour_pct = (
                (hour_used / quotas.requests_per_hour * 100) if quotas.requests_per_hour > 0 else 0
            )

            statuses["requests_per_hour"] = QuotaStatus(
                tenant_id=tenant_id,
                used=hour_used,
                remaining=hour_remaining,
                limit=quotas.requests_per_hour,
                reset_time=datetime.utcfromtimestamp(now + 3600),
                quota_type="requests_per_hour",
                is_exceeded=hour_remaining == 0,
                is_warning=hour_pct >= quotas.warn_threshold * 100,
                percentage_used=hour_pct,
            )

            # Per-day status
            day_timestamps = self._usage[tenant_id]["day"]
            day_cutoff = now - 86400
            day_timestamps[:] = [ts for ts in day_timestamps if ts > day_cutoff]
            day_used = len(day_timestamps)
            day_remaining = max(0, quotas.requests_per_day - day_used)
            day_pct = (
                (day_used / quotas.requests_per_day * 100) if quotas.requests_per_day > 0 else 0
            )

            statuses["requests_per_day"] = QuotaStatus(
                tenant_id=tenant_id,
                used=day_used,
                remaining=day_remaining,
                limit=quotas.requests_per_day,
                reset_time=datetime.utcfromtimestamp(now + 86400),
                quota_type="requests_per_day",
                is_exceeded=day_remaining == 0,
                is_warning=day_pct >= quotas.warn_threshold * 100,
                percentage_used=day_pct,
            )

            # Concurrent status
            concurrent_used = self._concurrent[tenant_id]
            concurrent_remaining = max(0, quotas.concurrent_requests - concurrent_used)
            concurrent_pct = (
                (concurrent_used / quotas.concurrent_requests * 100)
                if quotas.concurrent_requests > 0
                else 0
            )

            statuses["concurrent_requests"] = QuotaStatus(
                tenant_id=tenant_id,
                used=concurrent_used,
                remaining=concurrent_remaining,
                limit=quotas.concurrent_requests,
                reset_time=datetime.utcnow(),
                quota_type="concurrent_requests",
                is_exceeded=concurrent_remaining == 0,
                is_warning=concurrent_pct >= quotas.warn_threshold * 100,
                percentage_used=concurrent_pct,
            )

            return statuses

    async def reset(self, tenant_id: str) -> None:
        """
        Reset all quota usage for a tenant.

        Args:
            tenant_id: Tenant identifier.
        """
        async with self._lock:
            if tenant_id in self._usage:
                del self._usage[tenant_id]
            if tenant_id in self._concurrent:
                del self._concurrent[tenant_id]
            if tenant_id in self._bandwidth:
                del self._bandwidth[tenant_id]


__all__ = [
    "TenantQuotas",
    "QuotaStatus",
    "QuotaTracker",
]
