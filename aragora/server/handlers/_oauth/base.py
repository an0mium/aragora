"""
OAuth Handler - base class combining all provider mixins.

This is the main OAuthHandler class that inherits from SecureHandler and
all provider-specific mixins to handle OAuth authentication endpoints.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, cast
from collections.abc import Coroutine
from urllib.parse import urlencode

from aragora.rbac import AuthorizationContext, check_permission
from aragora.rbac.defaults import get_role_permissions

from aragora.server.handlers.base import HandlerResult, error_response
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.oauth.models import OAuthUserInfo

from .utils import _impl

# Maximum retries for database operations on connection errors
_DB_MAX_RETRIES = 3
_DB_RETRY_DELAY_BASE = 0.5  # seconds
from aragora.server.handlers.utils.rate_limit import get_client_ip

from .google import GoogleOAuthMixin
from .github import GitHubOAuthMixin
from .microsoft import MicrosoftOAuthMixin
from .apple import AppleOAuthMixin
from .oidc import OIDCOAuthMixin
from .account import AccountManagementMixin

logger = logging.getLogger(__name__)


def _maybe_await(value: Any) -> Any:
    """Resolve awaitables in sync OAuth handlers.

    Uses run_async() to dispatch to the main event loop where the asyncpg
    pool lives, instead of asyncio.run() which creates a temporary loop
    that is destroyed on return (breaking DB operations).
    """
    if inspect.isawaitable(value):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            from aragora.utils.async_utils import run_async

            return run_async(cast(Coroutine[Any, Any, Any], value))
    return value


class OAuthHandler(
    GoogleOAuthMixin,
    GitHubOAuthMixin,
    MicrosoftOAuthMixin,
    AppleOAuthMixin,
    OIDCOAuthMixin,
    AccountManagementMixin,
    SecureHandler,
):
    """Handler for OAuth authentication endpoints.

    Extends SecureHandler for JWT-based authentication, RBAC permission
    enforcement, and security audit logging.
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def _maybe_await(self, value: Any) -> Any:
        """Resolve awaitables for sync call sites."""
        return _maybe_await(value)

    async def _maybe_await_async(self, value: Any) -> Any:
        """Resolve awaitables for async call sites."""
        if inspect.isawaitable(value):
            return await value
        return value

    # Sync wrappers for async provider callbacks (used directly in tests).
    def _handle_google_callback(self, handler: Any, query_params: dict) -> HandlerResult:
        return self._maybe_await(super()._handle_google_callback(handler, query_params))

    def _handle_github_callback(self, handler: Any, query_params: dict) -> HandlerResult:
        return self._maybe_await(super()._handle_github_callback(handler, query_params))

    def _handle_microsoft_callback(self, handler: Any, query_params: dict) -> HandlerResult:
        return self._maybe_await(super()._handle_microsoft_callback(handler, query_params))

    def _handle_apple_callback(self, handler: Any, query_params: dict) -> HandlerResult:
        return self._maybe_await(super()._handle_apple_callback(handler, query_params))

    def _handle_oidc_callback(self, handler: Any, query_params: dict) -> HandlerResult:
        return self._maybe_await(super()._handle_oidc_callback(handler, query_params))

    def _handle_oidc_auth_start(self, handler: Any, query_params: dict) -> HandlerResult:
        return self._maybe_await(super()._handle_oidc_auth_start(handler, query_params))

    RESOURCE_TYPE = "oauth"

    # Support both v1 and non-v1 routes for backward compatibility
    ROUTES = [
        "/api/v1/auth/oauth/google",
        "/api/v1/auth/oauth/google/callback",
        "/api/v1/auth/oauth/github",
        "/api/v1/auth/oauth/github/callback",
        "/api/v1/auth/oauth/microsoft",
        "/api/v1/auth/oauth/microsoft/callback",
        "/api/v1/auth/oauth/apple",
        "/api/v1/auth/oauth/apple/callback",
        "/api/v1/auth/oauth/oidc",
        "/api/v1/auth/oauth/oidc/callback",
        "/api/v1/auth/oauth/url",
        "/api/v1/auth/oauth/authorize",
        "/api/v1/auth/oauth/callback",
        "/api/v1/auth/oauth/link",
        "/api/v1/auth/oauth/unlink",
        "/api/v1/auth/oauth/providers",
        "/api/v1/user/oauth-providers",
        # Non-v1 routes (for OAuth callback compatibility)
        "/api/auth/oauth/google",
        "/api/auth/oauth/google/callback",
        "/api/auth/oauth/github",
        "/api/auth/oauth/github/callback",
        "/api/auth/oauth/microsoft",
        "/api/auth/oauth/microsoft/callback",
        "/api/auth/oauth/apple",
        "/api/auth/oauth/apple/callback",
        "/api/auth/oauth/oidc",
        "/api/auth/oauth/oidc/callback",
        "/api/auth/oauth/url",
        "/api/auth/oauth/authorize",
        "/api/auth/oauth/callback",
        "/api/auth/oauth/link",
        "/api/auth/oauth/unlink",
        "/api/auth/oauth/providers",
        "/api/user/oauth-providers",
        # Diagnostic endpoint
        "/api/v1/auth/oauth/diagnostics",
        "/api/auth/oauth/diagnostics",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any, method: str = "GET"
    ) -> HandlerResult | None:
        """Route OAuth requests to appropriate methods."""
        # Extract provider from path for tracing
        provider = "unknown"
        if "/google" in path:
            provider = "google"
        elif "/github" in path:
            provider = "github"
        elif "/microsoft" in path:
            provider = "microsoft"
        elif "/apple" in path:
            provider = "apple"
        elif "/oidc" in path:
            provider = "oidc"

        # Access create_span/add_span_attributes via _impl() so test patches
        # applied to _oauth_impl.create_span are visible at runtime.
        impl = _impl()
        _cs = impl.create_span
        _asa = impl.add_span_attributes

        with _cs(f"oauth.{provider}", {"oauth.provider": provider, "oauth.path": path}) as span:
            # Get client IP for rate limiting
            client_ip = get_client_ip(handler)

            # Determine endpoint type for rate limiting:
            # - "token": Token exchange endpoints (POST /callback API, link/unlink)
            # - "callback": OAuth callback handlers (GET /callback from providers)
            # - "auth_start": Auth redirect endpoints (GET /google, /github, etc.)
            is_callback = "/callback" in path
            is_token_endpoint = (
                path.endswith("/link")
                or path.endswith("/unlink")
                or (is_callback and method == "POST")
            )

            if is_token_endpoint:
                endpoint_type = "token"
            elif is_callback:
                endpoint_type = "callback"
            else:
                endpoint_type = "auth_start"

            # Rate limit check using the wrapper in _impl (backward compatible)
            # The wrapper uses the new OAuthRateLimiter with exponential backoff
            if not impl._oauth_limiter.is_allowed(client_ip, endpoint_type):
                logger.warning(
                    "OAuth rate limit exceeded: ip=%s, endpoint=%s, provider=%s",
                    client_ip,
                    endpoint_type,
                    provider,
                )
                _asa(span, {"oauth.rate_limited": True})
                return error_response(
                    "Too many authentication attempts. Please try again later.",
                    429,
                )

            if hasattr(handler, "command"):
                method = handler.command

            # Add method to span
            _asa(span, {"oauth.method": method})

            # Normalize path - support both /api/v1/ and /api/ prefixes
            normalized = path.replace("/api/v1/", "/api/")

            # Determine if this is a callback (more important to trace)
            is_callback = "/callback" in normalized
            _asa(span, {"oauth.is_callback": is_callback})

            if normalized == "/api/auth/oauth/google" and method == "GET":
                return self._handle_google_auth_start(handler, query_params)

            if normalized == "/api/auth/oauth/google/callback" and method == "GET":
                return self._maybe_await(self._handle_google_callback(handler, query_params))

            if normalized == "/api/auth/oauth/github" and method == "GET":
                return self._handle_github_auth_start(handler, query_params)

            if normalized == "/api/auth/oauth/github/callback" and method == "GET":
                return self._maybe_await(self._handle_github_callback(handler, query_params))

            if normalized == "/api/auth/oauth/microsoft" and method == "GET":
                return self._handle_microsoft_auth_start(handler, query_params)

            if normalized == "/api/auth/oauth/microsoft/callback" and method == "GET":
                return self._maybe_await(self._handle_microsoft_callback(handler, query_params))

            if normalized == "/api/auth/oauth/apple" and method == "GET":
                return self._handle_apple_auth_start(handler, query_params)

            if normalized == "/api/auth/oauth/apple/callback" and method in ("GET", "POST"):
                return self._maybe_await(self._handle_apple_callback(handler, query_params))

            if normalized == "/api/auth/oauth/oidc" and method == "GET":
                return self._maybe_await(self._handle_oidc_auth_start(handler, query_params))

            if normalized == "/api/auth/oauth/oidc/callback" and method == "GET":
                return self._maybe_await(self._handle_oidc_callback(handler, query_params))

            if (
                normalized in ("/api/auth/oauth/url", "/api/auth/oauth/authorize")
                and method == "GET"
            ):
                return self._handle_oauth_url(handler, query_params)

            if normalized == "/api/auth/oauth/callback" and method == "POST":
                return self._handle_oauth_callback_api(handler)

            if normalized == "/api/auth/oauth/link" and method == "POST":
                return self._handle_link_account(handler)

            if normalized == "/api/auth/oauth/unlink" and method == "DELETE":
                return self._handle_unlink_account(handler)

            if normalized == "/api/auth/oauth/providers" and method == "GET":
                return self._handle_list_providers(handler)

            if normalized == "/api/user/oauth-providers" and method == "GET":
                return self._handle_get_user_providers(handler)

            if normalized == "/api/auth/oauth/diagnostics" and method == "GET":
                return self._handle_oauth_diagnostics(handler)

            _asa(span, {"oauth.error": "method_not_allowed"})
            return error_response("Method not allowed", 405)

    def _get_user_store(self) -> Any:
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _check_permission(
        self, handler: Any, permission_key: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        """Check RBAC permission. Returns error response if denied, None if allowed."""
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        # Not authenticated - return 401
        if not auth_ctx.is_authenticated or not auth_ctx.user_id:
            return error_response("Authentication required", 401)

        # Build RBAC authorization context
        roles = {auth_ctx.role} if auth_ctx.role else {"member"}
        permissions: set[str] = set()
        for role in roles:
            permissions |= get_role_permissions(role, include_inherited=True)

        rbac_context = AuthorizationContext(
            user_id=auth_ctx.user_id,
            org_id=auth_ctx.org_id,
            roles=roles,
            permissions=permissions,
            ip_address=auth_ctx.client_ip,
        )

        # Check permission
        decision = check_permission(rbac_context, permission_key, resource_id)
        if not decision.allowed:
            logger.warning(
                "Permission denied: user=%s permission=%s reason=%s",
                auth_ctx.user_id,
                permission_key,
                decision.reason,
            )
            return error_response("Permission denied", 403)

        return None  # Allowed

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def _handle_oauth_diagnostics(self, handler: Any) -> HandlerResult:
        """Return OAuth configuration diagnostics (admin-only).

        Requires admin:system permission to prevent information disclosure.
        """
        from aragora.server.handlers.base import json_response
        from aragora.server.handlers.oauth.config import get_oauth_config_status

        # Require admin permission
        perm_error = self._check_permission(handler, "admin:system")
        if perm_error is not None:
            return perm_error

        return json_response(get_oauth_config_status())

    # =========================================================================
    # Common OAuth Flow Completion
    # =========================================================================

    def _complete_oauth_flow(self, user_info: OAuthUserInfo, state_data: dict) -> HandlerResult:
        """Complete OAuth flow - create/login user and redirect with tokens."""
        return self._maybe_await(self._complete_oauth_flow_async(user_info, state_data))

    async def _complete_oauth_flow_async(
        self, user_info: OAuthUserInfo, state_data: dict
    ) -> HandlerResult:
        """Async implementation for completing OAuth flow."""
        user_store = self._get_user_store()
        if not user_store:
            return self._redirect_with_error("User service unavailable")

        # Check if this is account linking
        linking_user_id = state_data.get("user_id")
        if linking_user_id:
            return await self._maybe_await_async(
                self._handle_account_linking(user_store, linking_user_id, user_info, state_data)
            )

        # Check if user exists by OAuth provider ID
        user = await self._maybe_await_async(self._find_user_by_oauth(user_store, user_info))

        if not user:
            # Check if email already registered (use async to avoid nested loop)
            get_by_email = getattr(user_store, "get_user_by_email_async", None)
            if get_by_email and inspect.iscoroutinefunction(get_by_email):
                user = await get_by_email(user_info.email)
            else:
                user = user_store.get_user_by_email(user_info.email)
            if user:
                # Security: only link OAuth when email is verified by provider
                if not user_info.email_verified:
                    logger.warning(
                        "OAuth account linking blocked: unverified email %s from %s",
                        user_info.email,
                        user_info.provider,
                    )
                    return self._redirect_with_error(
                        "Email verification required to link your account."
                    )
                await self._maybe_await_async(
                    self._link_oauth_to_user(user_store, user.id, user_info)
                )
            else:
                user = await self._maybe_await_async(self._create_oauth_user(user_store, user_info))

        if not user:
            return self._redirect_with_error("Failed to create user account")

        # Update last login (use async to avoid nested loop)
        update_async = getattr(user_store, "update_user_async", None)
        if update_async and inspect.iscoroutinefunction(update_async):
            await update_async(user.id, last_login_at=datetime.now(timezone.utc))
        else:
            user_store.update_user(user.id, last_login_at=datetime.now(timezone.utc))

        # Create tokens
        from aragora.billing.jwt_auth import create_token_pair

        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=user.org_id,
            role=user.role,
        )

        logger.info("OAuth login: %s via %s", user.email, user_info.provider)

        redirect_url = state_data.get("redirect_url", _impl()._get_oauth_success_url())
        return self._redirect_with_tokens(redirect_url, tokens)

    def _find_user_by_oauth(self, user_store: Any, user_info: OAuthUserInfo) -> Any:
        """Find user by OAuth provider ID."""
        return self._maybe_await(self._find_user_by_oauth_async(user_store, user_info))

    async def _find_user_by_oauth_async(self, user_store: Any, user_info: OAuthUserInfo) -> Any:
        """Async implementation for finding user by OAuth provider ID.

        Includes retry logic for handling transient database connection errors
        (e.g., InterfaceError from stale asyncpg pools).
        """
        last_error: Exception | None = None

        for attempt in range(_DB_MAX_RETRIES):
            try:
                # Look for user with matching OAuth link
                # This requires the user store to support OAuth lookups
                async_lookup = getattr(user_store, "get_user_by_oauth_async", None)
                if async_lookup and inspect.iscoroutinefunction(async_lookup):
                    return await async_lookup(user_info.provider, user_info.provider_user_id)
                if hasattr(user_store, "get_user_by_oauth"):
                    return user_store.get_user_by_oauth(
                        user_info.provider, user_info.provider_user_id
                    )
                return None

            except Exception as e:  # noqa: BLE001 - DB driver exceptions (asyncpg.InterfaceError, psycopg2.Error) lack common importable base
                error_name = type(e).__name__
                # Check for retryable database connection errors
                is_retryable = error_name in (
                    "InterfaceError",  # asyncpg pool/connection invalid
                    "ConnectionDoesNotExistError",  # asyncpg connection closed
                    "ConnectionRefusedError",  # TCP connection refused
                    "TimeoutError",  # Connection timeout
                )
                if not is_retryable or attempt >= _DB_MAX_RETRIES - 1:
                    raise

                last_error = e
                delay = _DB_RETRY_DELAY_BASE * (2**attempt)
                logger.warning(
                    f"OAuth DB lookup failed (attempt {attempt + 1}/{_DB_MAX_RETRIES}): "
                    f"{error_name}: {e}. Retrying in {delay:.1f}s..."
                )

                # Try to refresh the pool before retrying
                try:
                    await self._try_refresh_user_store_pool(user_store)
                except (
                    ImportError,
                    ConnectionError,
                    OSError,
                    RuntimeError,
                    AttributeError,
                ) as refresh_err:
                    logger.warning("Pool refresh failed: %s", refresh_err)

                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        return None

    async def _try_refresh_user_store_pool(self, user_store: Any) -> None:
        """Attempt to refresh the user store's database pool.

        Force-reinitializes the shared pool to get fresh connections instead of
        reusing the same broken pool reference.
        """
        try:
            from aragora.storage.pool_manager import (
                get_shared_pool,
                initialize_shared_pool,
            )

            # Force-reinitialize the pool to get fresh connections.
            # Without force=True, initialize_shared_pool() returns the existing
            # (broken) pool since it's already initialized on this event loop.
            logger.info("Force-reinitializing shared pool after InterfaceError...")
            await initialize_shared_pool(force=True)

            # Get fresh pool reference
            new_pool = get_shared_pool()
            if new_pool and hasattr(user_store, "_pool"):
                user_store._pool = new_pool
                logger.info("User store pool reference updated with fresh pool")

        except ImportError:
            pass  # pool_manager not available

    def _link_oauth_to_user(self, user_store: Any, user_id: str, user_info: OAuthUserInfo) -> bool:
        """Link OAuth provider to existing user."""
        return self._maybe_await(self._link_oauth_to_user_async(user_store, user_id, user_info))

    async def _link_oauth_to_user_async(
        self, user_store: Any, user_id: str, user_info: OAuthUserInfo
    ) -> bool:
        """Async implementation for linking OAuth provider to existing user."""
        async_link = getattr(user_store, "link_oauth_provider_async", None)
        if async_link and inspect.iscoroutinefunction(async_link):
            return await async_link(
                user_id=user_id,
                provider=user_info.provider,
                provider_user_id=user_info.provider_user_id,
                email=user_info.email,
            )
        if hasattr(user_store, "link_oauth_provider"):
            return user_store.link_oauth_provider(
                user_id=user_id,
                provider=user_info.provider,
                provider_user_id=user_info.provider_user_id,
                email=user_info.email,
            )
        # Fallback: store in user metadata
        logger.warning("UserStore doesn't support OAuth linking, using fallback")
        return False

    def _create_oauth_user(self, user_store: Any, user_info: OAuthUserInfo) -> Any:
        """Create a new user from OAuth info."""
        return self._maybe_await(self._create_oauth_user_async(user_store, user_info))

    # OAuth providers whose login flow inherently verifies the email address.
    _TRUSTED_EMAIL_PROVIDERS = frozenset({"google", "microsoft", "apple", "github"})

    async def _create_oauth_user_async(self, user_store: Any, user_info: OAuthUserInfo) -> Any:
        """Async implementation for creating a new user from OAuth info."""
        # For trusted OAuth providers the authentication handshake itself proves
        # the user controls the email, so treat it as verified even when the
        # provider response omits or defaults the flag.
        if not user_info.email_verified:
            if user_info.provider in self._TRUSTED_EMAIL_PROVIDERS:
                logger.info(
                    "Auto-verifying email for trusted OAuth provider %s: %s",
                    user_info.provider,
                    user_info.email,
                )
            else:
                logger.warning(
                    "OAuth account creation blocked: unverified email %s from %s",
                    user_info.email,
                    user_info.provider,
                )
                return None

        from aragora.billing.models import hash_password

        # Generate random password (user will use OAuth to login)
        random_password = secrets.token_urlsafe(32)
        password_hash, password_salt = hash_password(random_password)

        try:
            create_async = getattr(user_store, "create_user_async", None)
            if create_async and inspect.iscoroutinefunction(create_async):
                user = await create_async(
                    email=user_info.email,
                    password_hash=password_hash,
                    password_salt=password_salt,
                    name=user_info.name,
                )
            else:
                user = user_store.create_user(
                    email=user_info.email,
                    password_hash=password_hash,
                    password_salt=password_salt,
                    name=user_info.name,
                )

            logger.debug("OAuth user created: id=%s, email=%s", user.id, user_info.email)

            # Link OAuth provider
            await self._link_oauth_to_user(user_store, user.id, user_info)  # type: ignore[misc]

            # Verify user was persisted (catches multi-worker SQLite, pool
            # refresh, and replication lag issues before we issue tokens).
            verify_async = getattr(user_store, "get_user_by_id_async", None)
            if verify_async and inspect.iscoroutinefunction(verify_async):
                verified = await verify_async(user.id)
            else:
                get_by_id = getattr(user_store, "get_user_by_id", None)
                verified = get_by_id(user.id) if callable(get_by_id) else None
            if not verified:
                logger.error(
                    "OAuth user created but NOT found on re-read: id=%s, email=%s, store=%s",
                    user.id,
                    user_info.email,
                    type(user_store).__name__,
                )

            # Log auto-provisioning with default role for audit trail
            logger.info(
                "OAuth user auto-provisioned: email=%s provider=%s user_id=%s role=%s action=rbac_auto_provision",
                user_info.email,
                user_info.provider,
                user.id,
                getattr(user, "role", "member"),
            )
            return user

        except ValueError as e:
            logger.error("Failed to create OAuth user: %s", e)
            return None

    def _handle_account_linking(
        self,
        user_store: Any,
        user_id: str,
        user_info: OAuthUserInfo,
        state_data: dict,
    ) -> HandlerResult:
        """Handle linking OAuth account to existing user."""
        return self._maybe_await(
            self._handle_account_linking_async(user_store, user_id, user_info, state_data)
        )

    async def _handle_account_linking_async(
        self,
        user_store: Any,
        user_id: str,
        user_info: OAuthUserInfo,
        state_data: dict,
    ) -> HandlerResult:
        """Async implementation for linking OAuth account to existing user."""
        # Verify user exists
        get_user_async = getattr(user_store, "get_user_by_id_async", None)
        if get_user_async and inspect.iscoroutinefunction(get_user_async):
            user = await get_user_async(user_id)
        else:
            user = user_store.get_user_by_id(user_id)
        if not user:
            return self._redirect_with_error("User not found")

        # Check if OAuth is already linked to another account
        existing_user = await self._maybe_await_async(
            self._find_user_by_oauth(user_store, user_info)
        )
        if existing_user and existing_user.id != user_id:
            return self._redirect_with_error(
                f"This {user_info.provider.title()} account is already linked to another user"
            )

        # Link OAuth
        success = await self._maybe_await_async(
            self._link_oauth_to_user(user_store, user_id, user_info)
        )
        if not success:
            logger.warning("OAuth linking fallback for user %s", user_id)

        redirect_url = state_data.get("redirect_url", _impl()._get_oauth_success_url())
        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=b"",
            headers={"Location": f"{redirect_url}?linked={user_info.provider}"},
        )

    # Cache-control headers to prevent CDN caching of OAuth redirects
    OAUTH_NO_CACHE_HEADERS = {
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    def _redirect_with_tokens(self, redirect_url: str, tokens: Any) -> HandlerResult:
        """Redirect to frontend with tokens in URL fragment.

        Tokens in query params can leak via logs, referrers, and proxies.
        Using a fragment keeps tokens client-side while allowing a 302 redirect.
        """
        params = urlencode(
            {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "token_type": "Bearer",
                "expires_in": tokens.expires_in,
            }
        )
        fragment_url = f"{redirect_url}#{params}"

        body = (
            f'<html><head><meta http-equiv="refresh" content="0;url={fragment_url}"></head>'
            f'<body>Redirecting... If you are not redirected, <a href="{fragment_url}">continue</a>.'
            f"</body></html>"
        )

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=body.encode(),
            headers={"Location": fragment_url, **self.OAUTH_NO_CACHE_HEADERS},
        )

    def _redirect_with_error(self, error: str) -> HandlerResult:
        """Redirect to error page with error message."""
        from urllib.parse import quote

        url = f"{_impl()._get_oauth_error_url()}?error={quote(error)}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={url}"></head></html>'.encode(),
            headers={"Location": url, **self.OAUTH_NO_CACHE_HEADERS},
        )
