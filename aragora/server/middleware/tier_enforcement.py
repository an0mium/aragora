"""
Tier Enforcement Middleware.

Enforces subscription tier limits on debate creation and other billable operations.
Returns 402 Payment Required when limits are exceeded.

Usage:
    from aragora.server.middleware.tier_enforcement import require_quota

    @require_quota("debate")
    def create_debate(self, handler):
        # Only called if org has available quota
        ...

    # Async usage with QuotaManager
    from aragora.server.middleware.tier_enforcement import get_quota_manager

    manager = get_quota_manager()
    if await manager.check_quota("debates", tenant_id=org_id):
        await manager.consume("debates", tenant_id=org_id)
"""

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.tenancy.quotas import QuotaManager

logger = logging.getLogger(__name__)

# Global QuotaManager singleton for unified quota enforcement
_quota_manager: Optional["QuotaManager"] = None


def get_quota_manager() -> "QuotaManager":
    """Get or create the global QuotaManager instance.

    Returns:
        QuotaManager instance for async quota operations.
    """
    global _quota_manager
    if _quota_manager is None:
        from aragora.tenancy.quotas import QuotaManager

        _quota_manager = QuotaManager()
    return _quota_manager


async def check_org_quota_async(
    org_id: str,
    resource: str = "debates",
) -> tuple[bool, Optional["QuotaExceededError"]]:
    """
    Async check if an organization has available quota.

    Uses the unified QuotaManager for tenant-aware quota checking.

    Args:
        org_id: Organization/tenant ID to check
        resource: Resource type (e.g., "debates", "api_requests")

    Returns:
        Tuple of (has_quota, error). If has_quota is False, error contains details.
    """
    if not org_id:
        return True, None

    try:
        manager = get_quota_manager()
        has_quota = await manager.check_quota(resource, tenant_id=org_id)
        if not has_quota:
            # Get current usage for error details
            status = await manager.get_quota_status(resource, tenant_id=org_id)
            if status:
                return False, QuotaExceededError(
                    resource=resource,
                    limit=status.limit,
                    used=status.current,
                    tier="unknown",  # QuotaManager doesn't track tier names
                    org_id=org_id,
                )
        return True, None
    except Exception as e:
        logger.error(f"Async quota check failed for org {org_id}: {e}")
        return True, None  # Fail open


async def increment_org_usage_async(
    org_id: str,
    resource: str = "debates",
    count: int = 1,
) -> bool:
    """
    Async increment usage counter using QuotaManager.

    Args:
        org_id: Organization/tenant ID
        resource: Resource type
        count: Amount to increment

    Returns:
        True if increment succeeded
    """
    if not org_id:
        return True

    try:
        manager = get_quota_manager()
        await manager.consume(resource, count, tenant_id=org_id)
        return True
    except Exception as e:
        logger.error(f"Async usage increment failed for org {org_id}: {e}")
        return False


class QuotaExceededError(Exception):
    """Raised when organization has exceeded their tier quota."""

    def __init__(
        self,
        resource: str,
        limit: int,
        used: int,
        tier: str,
        org_id: str,
    ):
        self.resource = resource
        self.limit = limit
        self.used = used
        self.tier = tier
        self.org_id = org_id
        self.remaining = max(0, limit - used)
        super().__init__(f"Quota exceeded for {resource}: {used}/{limit} ({tier} tier)")

    def to_response_dict(self) -> dict[str, Any]:
        """Convert to API response dictionary."""
        return {
            "error": "quota_exceeded",
            "code": "QUOTA_EXCEEDED",
            "message": f"Your {self.tier} plan allows {self.limit} {self.resource}s per month. "
            f"You have used {self.used}. Upgrade to increase your limit.",
            "resource": self.resource,
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "tier": self.tier,
            "upgrade_url": "/pricing",
        }


def check_org_quota(
    org_id: str,
    resource: str = "debate",
    user_store: Optional[Any] = None,
) -> tuple[bool, Optional[QuotaExceededError]]:
    """
    Check if an organization has available quota for a resource.

    Args:
        org_id: Organization ID to check
        resource: Resource type (currently only "debate" supported)
        user_store: Optional UserStore instance (uses global if not provided)

    Returns:
        Tuple of (has_quota, error). If has_quota is False, error contains details.
    """
    if not org_id:
        # No org = anonymous user, allow with free tier limits
        return True, None

    # Get user store
    if user_store is None:
        try:
            from aragora.storage.user_store import get_user_store

            user_store = get_user_store()
        except ImportError:
            logger.warning("UserStore not available, skipping quota check")
            return True, None

    try:
        org = user_store.get_organization_by_id(org_id)
        if org is None:
            logger.warning(f"Organization not found: {org_id}")
            return True, None  # Allow if org not found (shouldn't happen)

        if resource == "debate":
            if org.is_at_limit:
                return False, QuotaExceededError(
                    resource="debate",
                    limit=org.limits.debates_per_month,
                    used=org.debates_used_this_month,
                    tier=org.tier.value,
                    org_id=org_id,
                )

        return True, None

    except Exception as e:
        logger.error(f"Quota check failed for org {org_id}: {e}")
        # Fail open - don't block on quota check errors
        return True, None


def increment_org_usage(
    org_id: str,
    resource: str = "debate",
    count: int = 1,
    user_store: Optional[Any] = None,
) -> bool:
    """
    Increment usage counter for an organization.

    Should be called after successful resource creation.

    Args:
        org_id: Organization ID
        resource: Resource type
        count: Amount to increment
        user_store: Optional UserStore instance

    Returns:
        True if increment succeeded
    """
    if not org_id:
        return True

    if user_store is None:
        try:
            from aragora.storage.user_store import get_user_store

            user_store = get_user_store()
        except ImportError:
            return False

    if user_store is None:
        return False

    try:
        if resource == "debate":
            user_store.increment_usage(org_id, count)
            return True
        return True
    except Exception as e:
        logger.error(f"Failed to increment usage for org {org_id}: {e}")
        return False


def require_quota(resource: str = "debate") -> Callable:
    """
    Decorator that enforces tier quota limits.

    Returns 402 Payment Required if quota is exceeded.
    Automatically extracts org_id from authenticated user.

    Args:
        resource: Resource type to check quota for (default: "debate")

    Usage:
        @require_quota("debate")
        def create_debate(self, handler):
            ...

        @require_quota()  # defaults to "debate"
        def start_batch(self, handler):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from aragora.server.handlers.base import HandlerResult

            # Extract handler from args
            handler = kwargs.get("handler")
            if handler is None:
                for arg in args:
                    if hasattr(arg, "headers"):
                        handler = arg
                        break

            # Try to get org_id from authenticated user
            org_id = None
            user_store = None

            try:
                # Check handler class for user_store
                if handler is not None:
                    user_store = getattr(handler.__class__, "user_store", None)

                if user_store:
                    from aragora.billing.jwt_auth import extract_user_from_request

                    auth_ctx = extract_user_from_request(handler, user_store)
                    if auth_ctx.is_authenticated:
                        org_id = auth_ctx.org_id
            except Exception as e:
                logger.debug(f"Could not extract auth context: {e}")

            # Check quota
            has_quota, error = check_org_quota(org_id, resource, user_store)

            if not has_quota and error:
                logger.info(
                    f"Quota exceeded for org {org_id}: "
                    f"{error.used}/{error.limit} {resource}s ({error.tier} tier)"
                )
                return HandlerResult(
                    status_code=402,  # Payment Required
                    content_type="application/json",
                    body=error.to_response_dict(),
                )

            # Call the wrapped function
            result = func(*args, **kwargs)

            # Increment usage on success (status < 400)
            if isinstance(result, HandlerResult) and result.status_code < 400:
                increment_org_usage(org_id, resource, 1, user_store)

            return result

        return wrapper

    return decorator


def get_quota_status(
    org_id: str,
    user_store: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Get current quota status for an organization.

    Args:
        org_id: Organization ID
        user_store: Optional UserStore instance

    Returns:
        Dictionary with quota information
    """
    if not org_id:
        return {
            "tier": "free",
            "debates": {
                "limit": 10,
                "used": 0,
                "remaining": 10,
            },
            "is_at_limit": False,
        }

    if user_store is None:
        try:
            from aragora.storage.user_store import get_user_store

            user_store = get_user_store()
        except ImportError:
            return {"error": "UserStore not available"}

    try:
        org = user_store.get_organization_by_id(org_id)
        if org is None:
            return {"error": "Organization not found"}

        return {
            "tier": org.tier.value,
            "debates": {
                "limit": org.limits.debates_per_month,
                "used": org.debates_used_this_month,
                "remaining": org.debates_remaining,
            },
            "is_at_limit": org.is_at_limit,
            "billing_cycle_start": org.billing_cycle_start.isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get quota status for org {org_id}: {e}")
        return {"error": str(e)}


__all__ = [
    "QuotaExceededError",
    "check_org_quota",
    "increment_org_usage",
    "require_quota",
    "get_quota_status",
]
