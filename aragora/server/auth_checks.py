"""
Authentication and authorization checks for the unified server.

This module provides the AuthChecksMixin class with methods for:
- Rate limiting (basic API token, tier-based)
- RBAC permission checking
- Upload rate limiting

These methods are extracted from UnifiedHandler to improve modularity
and allow easier testing of authentication logic.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from aragora.server.middleware.rate_limit import RateLimitResult
    from aragora.storage import UserStore

logger = logging.getLogger(__name__)


class AuthChecksMixin:
    """Mixin providing authentication and authorization checking methods.

    This mixin expects the following class attributes from the parent:
    - headers: HTTP headers dict
    - command: HTTP method (GET, POST, etc.)
    - path: Request path
    - user_store: Optional UserStore for tier-based rate limiting

    And these methods from ResponseHelpersMixin:
    - _send_json(data, status): Send JSON response

    Auth exempt paths are defined as class attributes and can be customized
    by subclasses.
    """

    # Paths exempt from authentication (health checks, probes, OAuth flow, public read-only)
    AUTH_EXEMPT_PATHS: frozenset[str] = frozenset(
        [
            # Health checks (needed for load balancers, monitoring)
            "/healthz",
            "/readyz",
            "/api/health",
            "/api/health/detailed",
            "/api/health/deep",
            "/api/health/stores",
            "/api/v1/health",
            "/api/v1/health/detailed",
            "/api/v1/health/deep",
            "/api/v1/health/stores",
            # OAuth
            "/api/auth/oauth/providers",  # Login page needs to show available providers
            "/api/v1/auth/oauth/providers",  # v1 route
            # API documentation (public)
            "/api/openapi",
            "/api/openapi.json",
            "/api/openapi.yaml",
            "/api/postman.json",
            "/api/docs",
            "/api/docs/",
            "/api/redoc",
            "/api/redoc/",
            "/api/v1/openapi",
            "/api/v1/openapi.json",
            "/api/v1/docs",
            "/api/v1/docs/",
            # Read-only public endpoints
            "/api/insights/recent",
            "/api/flips/recent",
            "/api/evidence",
            "/api/evidence/statistics",
            "/api/verification/status",
            "/api/v1/insights/recent",
            "/api/v1/flips/recent",
            "/api/v1/evidence",
            "/api/v1/evidence/statistics",
            "/api/v1/verification/status",
            # Agent/ranking public data
            "/api/leaderboard",
            "/api/leaderboard-view",
            "/api/agents",
            "/api/v1/leaderboard",
            "/api/v1/leaderboard-view",
            "/api/v1/agents",
            # Public dashboard base paths (without trailing slash)
            # These match exact requests like /api/features that don't have a subpath
            "/api/features",
            "/api/v1/features",
            "/api/analytics",
            "/api/v1/analytics",
            "/api/replays",
            "/api/v1/replays",
            "/api/tournaments",
            "/api/v1/tournaments",
            "/api/reviews",
            "/api/v1/reviews",
            "/api/verticals",
            "/api/v1/verticals",
            "/api/evolution",
            "/api/v1/evolution",
            "/api/debates",
            "/api/v1/debates",
            "/api/moments",
            "/api/v1/moments",
            "/api/breakpoints",
            "/api/v1/breakpoints",
            "/api/consensus",
            "/api/v1/consensus",
            "/api/consensus/active",
            "/api/v1/consensus/active",
            "/api/pipeline/plans",
            "/api/v1/pipeline/plans",
            "/api/plans",
            "/api/v1/plans",
            # Metrics (public dashboard monitoring)
            "/api/metrics",
            "/api/v1/metrics",
            # Nomic state (public dashboard data)
            "/api/nomic/state",
            "/api/v1/nomic/state",
            # Playground - public demo endpoints (rate-limited by IP)
            "/api/v1/playground/debate",
            "/api/v1/playground/debate/live",
            "/api/v1/playground/debate/live/cost-estimate",
            "/api/v1/playground/status",
            "/api/v1/playground/tts",
        ]
    )

    # Path prefixes exempt from authentication (OAuth callbacks, read-only data)
    # Note: These endpoints bypass the legacy API token check (ARAGORA_API_TOKEN).
    # JWT authentication is still enforced via RBAC middleware for protected endpoints.
    AUTH_EXEMPT_PREFIXES: tuple[str, ...] = (
        "/api/auth/",  # All auth endpoints (JWT auth via RBAC, not API token)
        "/api/v1/auth/",  # All v1 auth endpoints (JWT auth via RBAC, not API token)
        "/api/agent/",  # Agent profiles (read-only)
        "/api/v1/agent/",  # Agent profiles v1 routes
        "/api/routing/",  # Domain detection and routing (read-only)
        "/api/v1/routing/",  # Domain routing v1 routes
    )

    # Path prefixes exempt ONLY for GET requests (read-only access)
    # These are public dashboard data endpoints that don't require auth for viewing
    AUTH_EXEMPT_GET_PREFIXES: tuple[str, ...] = (
        "/api/evidence/",  # Evidence read-only access
        "/api/v1/evidence/",  # Evidence read-only access (v1)
        "/api/evolution/",  # Public evolution data
        "/api/v1/evolution/",  # Public evolution data (v1)
        "/api/analytics/",  # Public analytics dashboards (stubbed when unauthenticated)
        "/api/v1/analytics/",  # Public analytics dashboards (v1)
        "/api/replays/",  # Public replay browsing
        "/api/v1/replays/",  # Public replay browsing (v1)
        "/api/learning/",  # Public learning evolution data
        "/api/v1/learning/",  # Public learning evolution data (v1)
        "/api/meta-learning/",  # Public meta-learning stats
        "/api/v1/meta-learning/",  # Public meta-learning stats (v1)
        "/api/tournaments/",  # Public tournament data
        "/api/v1/tournaments/",  # Public tournament data (v1)
        "/api/reviews/",  # Public reviews
        "/api/v1/reviews/",  # Public reviews (v1)
        "/api/consensus/",  # Public consensus read-only data
        "/api/v1/consensus/",  # Public consensus read-only data (v1)
        "/api/moments/",  # Public moments summaries
        "/api/v1/moments/",  # Public moments summaries (v1)
        "/api/flips/",  # Public flip summaries
        "/api/v1/flips/",  # Public flip summaries (v1)
        "/api/belief-network/",  # Public belief network summaries
        "/api/v1/belief-network/",  # Public belief network summaries (v1)
        "/api/verticals/",  # Public verticals list
        "/api/v1/verticals/",  # Public verticals list (v1)
        "/api/features/",  # Public feature config
        "/api/v1/features/",  # Public feature config (v1)
        "/api/gauntlet/personas",  # Public gauntlet personas list
        "/api/v1/gauntlet/personas",  # Public gauntlet personas list (v1)
        "/api/debates/",  # Public debate browsing
        "/api/v1/debates/",  # Public debate browsing (v1)
        "/api/metrics/",  # Public metrics dashboards
        "/api/v1/metrics/",  # Public metrics dashboards (v1)
        "/api/breakpoints/",  # Public breakpoints status
        "/api/v1/breakpoints/",  # Public breakpoints status (v1)
        "/api/plans/",  # Public decision plans
        "/api/v1/plans/",  # Public decision plans (v1)
        "/api/nomic/",  # Public nomic state
        "/api/v1/nomic/",  # Public nomic state (v1)
    )

    # Type stubs for attributes expected from parent class
    headers: Any
    command: str
    path: str
    user_store: Optional["UserStore"]
    rbac: Any

    # Per-request rate limit result (set by _check_tier_rate_limit)
    _rate_limit_result: Optional["RateLimitResult"] = None

    # Type stubs for methods expected from parent class
    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        """Send JSON response - provided by ResponseHelpersMixin."""
        ...

    def _is_path_exempt(self, path: str) -> bool:
        """Check if path is exempt from authentication.

        Args:
            path: The URL path to check

        Returns:
            True if the path is exempt from auth, False otherwise
        """
        if path in self.AUTH_EXEMPT_PATHS:
            return True
        if any(path.startswith(prefix) for prefix in self.AUTH_EXEMPT_PREFIXES):
            return True
        return False

    def _is_path_exempt_for_get(self, path: str) -> bool:
        """Check if path is exempt from authentication for GET requests only.

        Args:
            path: The URL path to check

        Returns:
            True if the path is exempt for GET, False otherwise
        """
        return any(path.startswith(prefix) for prefix in self.AUTH_EXEMPT_GET_PREFIXES)

    def _check_rate_limit(self) -> bool:
        """Check auth and rate limit. Returns True if allowed, False if blocked.

        Sends appropriate error response if blocked.

        This method checks:
        1. If path is exempt from auth
        2. If auth is enabled
        3. If authentication is valid
        4. If rate limit is exceeded

        Returns:
            True if request is allowed, False if blocked
        """
        # Check exemptions BEFORE imports so exempt paths work even if
        # auth modules have import issues in production
        parsed = urlparse(self.path)
        if self._is_path_exempt(parsed.path):
            return True
        # For GET-only exempt paths, check method
        if self.command == "GET" and self._is_path_exempt_for_get(parsed.path):
            return True

        from aragora.server.auth import auth_config, check_auth

        if not auth_config.enabled:
            return True

        # Convert headers to dict
        headers = {k: v for k, v in self.headers.items()}

        authenticated, remaining = check_auth(headers, parsed.query)

        if not authenticated:
            if remaining == 0:
                # Rate limited
                self._send_json({"error": "Rate limit exceeded. Try again later."}, status=429)
            else:
                # Auth failed
                self._send_json({"error": "Authentication required"}, status=401)
            return False

        # Note: Rate limit headers are now added by individual handlers
        # that need to include them in their responses
        return True

    def _check_tier_rate_limit(self) -> bool:
        """Check tier-aware rate limit based on user's subscription.

        Returns True if allowed, False if blocked.
        Sends 429 error response if rate limited.
        Also stores the result for inclusion in response headers.

        The tier-based rate limiting applies different limits based on
        the user's subscription tier (free, pro, enterprise, etc.).

        Returns:
            True if request is allowed, False if rate limited
        """
        from aragora.server.middleware.rate_limit import check_tier_rate_limit

        result = check_tier_rate_limit(self, self.user_store)

        # Store result for response headers (used by _add_rate_limit_headers)
        self._rate_limit_result = result

        if not result.allowed:
            self._send_json(
                {
                    "error": "Rate limit exceeded for your subscription tier",
                    "code": "tier_rate_limit",
                    "limit": result.limit,
                    "retry_after": int(result.retry_after) + 1,
                    "upgrade_url": "/pricing",
                },
                status=429,
            )
            return False

        return True

    def _check_rbac(self, path: str, method: str) -> bool:
        """Check RBAC permission for the request.

        Returns True if allowed, False if blocked.
        Sends 401/403 error response if denied.

        This method:
        1. Checks if path is exempt from RBAC
        2. Extracts user context from JWT
        3. Builds authorization context with roles and permissions
        4. Checks if user has required permission for the route

        Args:
            path: The URL path being accessed
            method: The HTTP method (GET, POST, etc.)

        Returns:
            True if request is allowed, False if denied
        """
        # Check exemptions BEFORE imports so exempt paths work even if
        # billing/rbac modules have import issues
        if self._is_path_exempt(path):
            return True
        if method.upper() == "GET" and self._is_path_exempt_for_get(path):
            return True

        # When authentication is disabled (no ARAGORA_API_TOKEN), allow all
        # requests through RBAC — there's no user to authorize against.
        from aragora.server.auth import auth_config

        if not auth_config.enabled:
            return True

        from aragora.billing.auth import extract_user_from_request
        from aragora.rbac import AuthorizationContext, get_role_permissions

        logger.debug("RBAC auth check: %s %s", method, path)

        # Build authorization context from JWT
        auth_ctx = None
        try:
            user_ctx = extract_user_from_request(self, self.user_store)
            logger.debug(
                "RBAC user context: authenticated=%s, user_id=%s",
                user_ctx.authenticated,
                user_ctx.user_id,
            )
            if user_ctx.authenticated and user_ctx.user_id:
                roles = {user_ctx.role} if user_ctx.role else {"member"}
                permissions: set[str] = set()
                for role in roles:
                    permissions |= get_role_permissions(role, include_inherited=True)

                auth_ctx = AuthorizationContext(
                    user_id=user_ctx.user_id,
                    org_id=user_ctx.org_id,
                    roles=roles,
                    permissions=permissions,
                    ip_address=user_ctx.client_ip,
                )
                logger.debug("RBAC auth context created for user %s", user_ctx.user_id)
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.debug("RBAC context extraction failed: %s", e)

        # Check permission
        allowed, reason, permission_key = self.rbac.check_request(path, method, auth_ctx)

        if not allowed:
            if auth_ctx is None:
                self._send_json(
                    {"error": "Authentication required", "code": "auth_required"},
                    status=401,
                )
            else:
                self._send_json(
                    {
                        "error": f"Permission denied: {reason}",
                        "code": "permission_denied",
                        "required_permission": permission_key,
                    },
                    status=403,
                )
            return False

        return True

    def _check_upload_rate_limit(self) -> bool:
        """Check IP-based upload rate limit. Returns True if allowed, False if blocked.

        This is a separate rate limit for file uploads to prevent abuse.
        Uses IP-based limiting regardless of authentication status.

        Returns:
            True if upload is allowed, False if rate limited
        """
        from typing import cast
        from http.server import BaseHTTPRequestHandler

        from aragora.server.upload_rate_limit import get_upload_limiter

        limiter = get_upload_limiter()
        # Cast self to BaseHTTPRequestHandler for type checking - the mixin
        # expects to be mixed with a handler that has client_address and headers
        client_ip = limiter.get_client_ip(cast(BaseHTTPRequestHandler, self))
        allowed, error_info = limiter.check_allowed(client_ip)

        if not allowed and error_info:
            self._send_json(
                {"error": error_info["message"], "retry_after": error_info["retry_after"]},
                status=429,
            )
            return False

        return True


__all__ = ["AuthChecksMixin"]
