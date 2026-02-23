"""
Tier-Based Feature Gating.

Provides the @require_tier decorator to enforce subscription-tier restrictions
on handler methods. Free-tier users are blocked from premium features with a
clear upgrade prompt.

Usage:
    from aragora.billing.tier_gating import require_tier

    @require_tier("professional")
    async def knowledge_mound_query(context, ...):
        ...

    @require_tier("enterprise")
    async def saml_sso_login(context, ...):
        ...

Tier hierarchy (lowest to highest):
    free < starter < professional < enterprise < enterprise_plus

The decorator extracts the user's organization tier from the AuthorizationContext
(via org_id -> Organization lookup). If the billing module is unavailable or the
org cannot be resolved, access is allowed (graceful degradation).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, TypeVar, ParamSpec, cast
from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# --- Tier hierarchy (ordinal for comparison) --------------------------------

TIER_ORDER: dict[str, int] = {
    "free": 0,
    "starter": 1,
    "professional": 2,
    "enterprise": 3,
    "enterprise_plus": 4,
}

# Friendly display names for upgrade prompts
TIER_DISPLAY_NAMES: dict[str, str] = {
    "free": "Free",
    "starter": "Starter",
    "professional": "Professional",
    "enterprise": "Enterprise",
    "enterprise_plus": "Enterprise+",
}


# --- Feature -> minimum tier mapping ----------------------------------------

FEATURE_TIER_MAP: dict[str, str] = {
    # Knowledge Mound - requires Professional
    "knowledge_mound": "professional",
    # Slack/Teams integrations - requires Professional
    "slack_integration": "professional",
    "teams_integration": "professional",
    # Unified memory / Supermemory - requires Professional
    "unified_memory": "professional",
    # SSO (SAML/OIDC) - requires Enterprise
    "sso": "enterprise",
    # Audit log access - requires Enterprise
    "audit_logs": "enterprise",
    # Custom agents - requires Enterprise
    "custom_agents": "enterprise",
}


# --- Tier comparison helper --------------------------------------------------

def tier_sufficient(user_tier: str, required_tier: str) -> bool:
    """Check whether user_tier meets or exceeds required_tier."""
    user_rank = TIER_ORDER.get(user_tier, 0)
    required_rank = TIER_ORDER.get(required_tier, 0)
    return user_rank >= required_rank


# --- Organization tier resolver ----------------------------------------------

def _resolve_org_tier(context: Any) -> str | None:
    """Resolve the subscription tier for a user's organization.

    Attempts to look up the Organization from billing stores. Returns the
    tier string (e.g. "free", "professional") or None if resolution fails.
    """
    org_id = getattr(context, "org_id", None)
    if not org_id or not isinstance(org_id, str):
        return None

    try:
        from aragora.billing.models import Organization  # noqa: F401 — availability check

        # Try to get org from server-side store (if available)
        try:
            from aragora.storage.repositories.org import get_org_repository

            repo = get_org_repository()
            org = repo.get(org_id)
            if org and hasattr(org, "tier"):
                tier_val = org.tier.value if hasattr(org.tier, "value") else str(org.tier)
                if isinstance(tier_val, str) and tier_val in TIER_ORDER:
                    return tier_val
        except (ImportError, AttributeError, RuntimeError, TypeError):
            pass

        # Fallback: check if context itself carries tier metadata
        tier = getattr(context, "subscription_tier", None)
        if tier is not None:
            tier_str = tier.value if hasattr(tier, "value") else str(tier)
            if isinstance(tier_str, str) and tier_str in TIER_ORDER:
                return tier_str

    except ImportError:
        pass

    return None


# --- Error class for tier violations ----------------------------------------

class TierInsufficientError(Exception):
    """Raised when a user's subscription tier is too low for a feature."""

    def __init__(
        self,
        required_tier: str,
        current_tier: str,
        feature: str | None = None,
    ) -> None:
        self.required_tier = required_tier
        self.current_tier = current_tier
        self.feature = feature
        display_required = TIER_DISPLAY_NAMES.get(required_tier, required_tier)
        msg = f"This feature requires the {display_required} plan"
        if feature:
            msg = f"{feature} requires the {display_required} plan"
        super().__init__(msg)

    def to_response(self) -> dict[str, Any]:
        """Build a JSON-serializable error response with upgrade prompt."""
        display_required = TIER_DISPLAY_NAMES.get(
            self.required_tier, self.required_tier
        )
        return {
            "error": str(self),
            "code": "tier_insufficient",
            "required_tier": self.required_tier,
            "current_tier": self.current_tier,
            "upgrade_url": "/pricing",
            "upgrade_prompt": f"Upgrade to {display_required} to unlock this feature.",
        }


# --- Decorator ---------------------------------------------------------------

def require_tier(
    minimum_tier: str,
    *,
    feature_name: str | None = None,
    context_param: str = "context",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to require a minimum subscription tier.

    Args:
        minimum_tier: Minimum tier needed (e.g. "professional", "enterprise").
        feature_name: Optional human-readable feature name for error messages.
        context_param: Parameter name for AuthorizationContext (default "context").

    Behaviour:
        - Extracts AuthorizationContext from function args/kwargs.
        - Resolves the organization's tier.
        - If tier is insufficient, raises TierInsufficientError.
        - If billing/org cannot be resolved (graceful degradation), allows access.
    """
    if minimum_tier not in TIER_ORDER:
        raise ValueError(
            f"Unknown tier {minimum_tier!r}. "
            f"Valid tiers: {list(TIER_ORDER.keys())}"
        )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:

        def _is_auth_context(obj: Any) -> bool:
            """Check if obj is an AuthorizationContext (by type or duck-typing)."""
            try:
                from aragora.rbac.models import AuthorizationContext
                if isinstance(obj, AuthorizationContext):
                    return True
            except ImportError:
                pass
            # Duck-type: must have user_id (str) and optionally org_id (str or None)
            user_id = getattr(obj, "user_id", None)
            if not isinstance(user_id, str):
                return False
            org_id = getattr(obj, "org_id", None)
            return org_id is None or isinstance(org_id, str)

        def _get_context(
            args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> Any | None:
            """Extract AuthorizationContext from arguments."""
            # Check kwargs
            if context_param in kwargs:
                ctx = kwargs[context_param]
                if _is_auth_context(ctx):
                    return ctx
            # Check positional args
            for arg in args:
                if _is_auth_context(arg):
                    return arg
            # Check if handler has _auth_context attribute
            for arg in args:
                ctx = getattr(arg, "_auth_context", None)
                if ctx is not None and _is_auth_context(ctx):
                    return ctx
            return None

        def _check_tier(context: Any) -> None:
            """Validate tier, raise TierInsufficientError if insufficient."""
            org_tier = _resolve_org_tier(context)
            if org_tier is None:
                # Graceful degradation: if we cannot determine tier, allow access
                logger.debug(
                    "Could not resolve org tier for user %s; allowing access",
                    getattr(context, "user_id", "unknown"),
                )
                return
            if not tier_sufficient(org_tier, minimum_tier):
                raise TierInsufficientError(
                    required_tier=minimum_tier,
                    current_tier=org_tier,
                    feature=feature_name,
                )

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context(args, kwargs)
            if context is not None:
                _check_tier(context)
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context(args, kwargs)
            if context is not None:
                _check_tier(context)
            return await cast(Coroutine[Any, Any, T], func(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, T], async_wrapper)
        return cast(Callable[P, T], sync_wrapper)

    return decorator


# --- Free-tier debate rate limiter -------------------------------------------

class DebateRateLimiter:
    """Track debate usage per organization for free-tier enforcement.

    This is a simple in-memory counter. Production deployments should
    back this with the Organization.debates_used_this_month field in the
    database (already tracked by Organization.increment_debates()).
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def check_and_increment(self, org_id: str, tier: str) -> dict[str, Any] | None:
        """Check debate limit and increment counter.

        Returns None if allowed, or an error dict if limit exceeded.
        """
        try:
            from aragora.billing.models import TIER_LIMITS, SubscriptionTier

            tier_enum = SubscriptionTier(tier)
            limits = TIER_LIMITS[tier_enum]
            max_debates = limits.debates_per_month
        except (ImportError, ValueError, KeyError):
            # Cannot resolve limits; allow access
            return None

        current = self._counts.get(org_id, 0)
        if current >= max_debates:
            display = TIER_DISPLAY_NAMES.get("professional", "Professional")
            return {
                "error": f"Monthly debate limit ({max_debates}) reached for your plan.",
                "code": "debate_limit_exceeded",
                "current_usage": current,
                "limit": max_debates,
                "upgrade_url": "/pricing",
                "upgrade_prompt": f"Upgrade to {display} for more debates.",
            }
        self._counts[org_id] = current + 1
        return None

    def reset(self, org_id: str) -> None:
        """Reset debate count for an organization (e.g. at billing cycle)."""
        self._counts.pop(org_id, None)

    def get_usage(self, org_id: str) -> int:
        """Get current debate count for an organization."""
        return self._counts.get(org_id, 0)


# Singleton for convenience
_debate_limiter: DebateRateLimiter | None = None


def get_debate_rate_limiter() -> DebateRateLimiter:
    """Get the global debate rate limiter instance."""
    global _debate_limiter
    if _debate_limiter is None:
        _debate_limiter = DebateRateLimiter()
    return _debate_limiter


__all__ = [
    "TIER_ORDER",
    "TIER_DISPLAY_NAMES",
    "FEATURE_TIER_MAP",
    "tier_sufficient",
    "TierInsufficientError",
    "require_tier",
    "DebateRateLimiter",
    "get_debate_rate_limiter",
]
