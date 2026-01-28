"""
Tenant resource limits enforcement.

Provides quota and rate limit checking for tenant resources.

Usage:
    from aragora.tenancy.limits import TenantLimitsEnforcer
    from aragora.tenancy.tenant import TenantConfig

    config = TenantConfig(max_debates_per_day=10, tokens_per_month=100000)
    enforcer = TenantLimitsEnforcer(config)

    await enforcer.check_debate_limit("tenant-1", current_count=5)
    await enforcer.check_token_budget("tenant-1", tokens_used=50000, tokens_requested=1000)
"""

from __future__ import annotations

import logging
from typing import Optional

from aragora.tenancy.tenant import TenantConfig

logger = logging.getLogger(__name__)


class TenantLimitExceededError(Exception):
    """Raised when a tenant exceeds their resource limits."""

    def __init__(
        self,
        message: str,
        limit_type: str,
        current: int,
        limit: int,
        tenant_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.tenant_id = tenant_id


class TenantLimitsEnforcer:
    """
    Enforces tenant resource limits.

    Checks quotas for debates, tokens, storage, and other resources.
    """

    def __init__(self, config: TenantConfig):
        """Initialize the enforcer.

        Args:
            config: Tenant configuration with limits
        """
        self.config = config

    async def check_debate_limit(
        self,
        tenant_id: str,
        current_count: int,
    ) -> bool:
        """Check if tenant can create another debate.

        Args:
            tenant_id: The tenant ID
            current_count: Current number of debates today

        Returns:
            True if allowed

        Raises:
            TenantLimitExceededError: If limit exceeded
        """
        if current_count >= self.config.max_debates_per_day:
            raise TenantLimitExceededError(
                f"Daily debate limit exceeded: {current_count}/{self.config.max_debates_per_day}",
                limit_type="debates_per_day",
                current=current_count,
                limit=self.config.max_debates_per_day,
                tenant_id=tenant_id,
            )
        return True

    async def check_token_budget(
        self,
        tenant_id: str,
        tokens_used: int,
        tokens_requested: int,
    ) -> bool:
        """Check if tenant has enough token budget.

        Args:
            tenant_id: The tenant ID
            tokens_used: Tokens already used this month
            tokens_requested: Tokens needed for current operation

        Returns:
            True if allowed

        Raises:
            TenantLimitExceededError: If budget exceeded
        """
        total = tokens_used + tokens_requested
        if total > self.config.tokens_per_month:
            raise TenantLimitExceededError(
                f"Monthly token limit exceeded: {total}/{self.config.tokens_per_month}",
                limit_type="tokens_per_month",
                current=tokens_used,
                limit=self.config.tokens_per_month,
                tenant_id=tenant_id,
            )
        return True

    async def check_storage_quota(
        self,
        tenant_id: str,
        current_usage: int,
        bytes_requested: int,
    ) -> bool:
        """Check if tenant has enough storage quota.

        Args:
            tenant_id: The tenant ID
            current_usage: Current storage usage in bytes
            bytes_requested: Bytes needed for new data

        Returns:
            True if allowed

        Raises:
            TenantLimitExceededError: If quota exceeded
        """
        total = current_usage + bytes_requested
        if total > self.config.storage_quota:
            raise TenantLimitExceededError(
                f"Storage quota exceeded: {total}/{self.config.storage_quota} bytes",
                limit_type="storage_quota",
                current=current_usage,
                limit=self.config.storage_quota,
                tenant_id=tenant_id,
            )
        return True

    async def check_concurrent_debates(
        self,
        tenant_id: str,
        active_count: int,
    ) -> bool:
        """Check if tenant can run another concurrent debate.

        Args:
            tenant_id: The tenant ID
            active_count: Number of currently active debates

        Returns:
            True if allowed

        Raises:
            TenantLimitExceededError: If limit exceeded
        """
        if active_count >= self.config.max_concurrent_debates:
            raise TenantLimitExceededError(
                f"Concurrent debate limit exceeded: {active_count}/{self.config.max_concurrent_debates}",
                limit_type="concurrent_debates",
                current=active_count,
                limit=self.config.max_concurrent_debates,
                tenant_id=tenant_id,
            )
        return True

    async def check_api_rate_limit(
        self,
        tenant_id: str,
        requests_this_minute: int,
    ) -> bool:
        """Check if tenant is within API rate limits.

        Args:
            tenant_id: The tenant ID
            requests_this_minute: Number of requests in current minute

        Returns:
            True if allowed

        Raises:
            TenantLimitExceededError: If rate limit exceeded
        """
        if requests_this_minute >= self.config.api_requests_per_minute:
            raise TenantLimitExceededError(
                f"API rate limit exceeded: {requests_this_minute}/{self.config.api_requests_per_minute} req/min",
                limit_type="api_rate_limit",
                current=requests_this_minute,
                limit=self.config.api_requests_per_minute,
                tenant_id=tenant_id,
            )
        return True

    def get_usage_summary(
        self,
        debates_today: int = 0,
        tokens_this_month: int = 0,
        storage_bytes: int = 0,
        active_debates: int = 0,
    ) -> dict:
        """Get a summary of usage vs limits.

        Args:
            debates_today: Debates created today
            tokens_this_month: Tokens used this month
            storage_bytes: Current storage usage
            active_debates: Currently active debates

        Returns:
            Dictionary with usage percentages and remaining quotas
        """
        return {
            "debates": {
                "used": debates_today,
                "limit": self.config.max_debates_per_day,
                "remaining": max(0, self.config.max_debates_per_day - debates_today),
                "percentage": round(debates_today / self.config.max_debates_per_day * 100, 1),
            },
            "tokens": {
                "used": tokens_this_month,
                "limit": self.config.tokens_per_month,
                "remaining": max(0, self.config.tokens_per_month - tokens_this_month),
                "percentage": round(tokens_this_month / self.config.tokens_per_month * 100, 1),
            },
            "storage": {
                "used": storage_bytes,
                "limit": self.config.storage_quota,
                "remaining": max(0, self.config.storage_quota - storage_bytes),
                "percentage": round(storage_bytes / self.config.storage_quota * 100, 1),
            },
            "concurrent_debates": {
                "active": active_debates,
                "limit": self.config.max_concurrent_debates,
                "remaining": max(0, self.config.max_concurrent_debates - active_debates),
            },
        }
