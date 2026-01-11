"""
Aragora Server Middleware Package.

Provides composable middleware decorators for common cross-cutting concerns:
- Authentication (require_auth, optional_auth)
- Rate limiting (rate_limit, RateLimiter)
- Caching (cache, ttl_cache)

Usage:
    from aragora.server.middleware import require_auth, rate_limit, cache

    @require_auth
    @rate_limit(requests_per_minute=30)
    @cache(ttl_seconds=300)
    def get_leaderboard(self, handler):
        ...

Middleware can be stacked in any order, but recommended order is:
1. Authentication (first - reject unauthorized requests early)
2. Rate limiting (second - prevent abuse before expensive operations)
3. Caching (last - serve cached responses when available)
"""

from .auth import (
    require_auth,
    require_auth_or_localhost,
    optional_auth,
    extract_token,
    validate_token,
    extract_client_ip,
    AuthContext,
)
from .rate_limit import (
    rate_limit,
    RateLimiter,
    get_rate_limiter,
    cleanup_rate_limiters,
    TierRateLimiter,
    get_tier_rate_limiter,
    check_tier_rate_limit,
    TIER_RATE_LIMITS,
)
from .cache import (
    cache,
    ttl_cache,
    clear_cache,
    invalidate_cache,
    get_cache_stats,
    CacheConfig,
    CACHE_INVALIDATION_MAP,
)
from .request_logging import (
    REQUEST_ID_HEADER,
    RequestContext,
    generate_request_id,
    hash_token,
    sanitize_headers,
    sanitize_params,
    log_request,
    log_response,
    request_logging,
    get_current_request_id,
    set_current_request_id,
)
from .auth_v2 import (
    User,
    Workspace,
    APIKey,
    require_user,
    require_admin,
    require_plan,
    get_current_user,
    authenticate_request,
    SupabaseAuthValidator,
    get_jwt_validator,
)
from .tenancy import (
    PLAN_LIMITS,
    get_plan_limits,
    WorkspaceManager,
    get_workspace_manager,
    require_workspace,
    check_limit,
    tenant_scoped,
    scope_query,
    ensure_workspace_access,
)

__all__ = [
    # Auth
    "require_auth",
    "require_auth_or_localhost",
    "optional_auth",
    "extract_token",
    "validate_token",
    "extract_client_ip",
    "AuthContext",
    # Rate limiting
    "rate_limit",
    "RateLimiter",
    "get_rate_limiter",
    "cleanup_rate_limiters",
    "TierRateLimiter",
    "get_tier_rate_limiter",
    "check_tier_rate_limit",
    "TIER_RATE_LIMITS",
    # Caching
    "cache",
    "ttl_cache",
    "clear_cache",
    "invalidate_cache",
    "get_cache_stats",
    "CacheConfig",
    "CACHE_INVALIDATION_MAP",
    # Request logging
    "REQUEST_ID_HEADER",
    "RequestContext",
    "generate_request_id",
    "hash_token",
    "sanitize_headers",
    "sanitize_params",
    "log_request",
    "log_response",
    "request_logging",
    "get_current_request_id",
    "set_current_request_id",
    # User Auth (Supabase)
    "User",
    "Workspace",
    "APIKey",
    "require_user",
    "require_admin",
    "require_plan",
    "get_current_user",
    "authenticate_request",
    "SupabaseAuthValidator",
    "get_jwt_validator",
    # Multi-tenancy
    "PLAN_LIMITS",
    "get_plan_limits",
    "WorkspaceManager",
    "get_workspace_manager",
    "require_workspace",
    "check_limit",
    "tenant_scoped",
    "scope_query",
    "ensure_workspace_access",
]
