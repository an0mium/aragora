"""
OAuth Handler - base class combining all provider mixins.

This is the main OAuthHandler class that inherits from SecureHandler and
all provider-specific mixins to handle OAuth authentication endpoints.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from urllib.parse import urlencode

from aragora.rbac import AuthorizationContext, check_permission
from aragora.rbac.defaults import get_role_permissions

from aragora.server.handlers.base import HandlerResult, error_response
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.oauth.models import OAuthUserInfo

from .utils import _impl
from aragora.server.handlers.utils.rate_limit import get_client_ip

from .google import GoogleOAuthMixin
from .github import GitHubOAuthMixin
from .microsoft import MicrosoftOAuthMixin
from .apple import AppleOAuthMixin
from .oidc import OIDCOAuthMixin
from .account import AccountManagementMixin

logger = logging.getLogger(__name__)


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
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict, handler, method: str = "GET"
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
        _cs = impl.create_span  # noqa: F841
        _asa = impl.add_span_attributes  # noqa: F841

        with _cs(f"oauth.{provider}", {"oauth.provider": provider, "oauth.path": path}) as span:
            # Rate limit check
            client_ip = get_client_ip(handler)
            if not impl._oauth_limiter.is_allowed(client_ip):
                logger.warning(f"Rate limit exceeded for OAuth endpoint: {client_ip}")
                _asa(span, {"oauth.rate_limited": True})
                return error_response("Rate limit exceeded. Please try again later.", 429)

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
                return self._handle_google_callback(handler, query_params)

            if normalized == "/api/auth/oauth/github" and method == "GET":
                return self._handle_github_auth_start(handler, query_params)

            if normalized == "/api/auth/oauth/github/callback" and method == "GET":
                return self._handle_github_callback(handler, query_params)

            if normalized == "/api/auth/oauth/microsoft" and method == "GET":
                return self._handle_microsoft_auth_start(handler, query_params)

            if normalized == "/api/auth/oauth/microsoft/callback" and method == "GET":
                return self._handle_microsoft_callback(handler, query_params)

            if normalized == "/api/auth/oauth/apple" and method == "GET":
                return self._handle_apple_auth_start(handler, query_params)

            if normalized == "/api/auth/oauth/apple/callback" and method in ("GET", "POST"):
                return self._handle_apple_callback(handler, query_params)

            if normalized == "/api/auth/oauth/oidc" and method == "GET":
                return self._handle_oidc_auth_start(handler, query_params)

            if normalized == "/api/auth/oauth/oidc/callback" and method == "GET":
                return self._handle_oidc_callback(handler, query_params)

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

            _asa(span, {"oauth.error": "method_not_allowed"})
            return error_response("Method not allowed", 405)

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _check_permission(
        self, handler, permission_key: str, resource_id: str | None = None
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
                f"Permission denied: user={auth_ctx.user_id} permission={permission_key} reason={decision.reason}"
            )
            return error_response(f"Permission denied: {decision.reason}", 403)

        return None  # Allowed

    # =========================================================================
    # Common OAuth Flow Completion
    # =========================================================================

    async def _complete_oauth_flow(
        self, user_info: OAuthUserInfo, state_data: dict
    ) -> HandlerResult:
        """Complete OAuth flow - create/login user and redirect with tokens."""
        user_store = self._get_user_store()
        if not user_store:
            return self._redirect_with_error("User service unavailable")

        # Check if this is account linking
        linking_user_id = state_data.get("user_id")
        if linking_user_id:
            return await self._handle_account_linking(
                user_store, linking_user_id, user_info, state_data
            )

        # Check if user exists by OAuth provider ID
        user = await self._find_user_by_oauth(user_store, user_info)

        if not user:
            # Check if email already registered
            user = user_store.get_user_by_email(user_info.email)
            if user:
                await self._link_oauth_to_user(user_store, user.id, user_info)
            else:
                user = await self._create_oauth_user(user_store, user_info)

        if not user:
            return self._redirect_with_error("Failed to create user account")

        # Update last login
        user_store.update_user(user.id, last_login_at=datetime.now(timezone.utc))

        # Create tokens
        from aragora.billing.jwt_auth import create_token_pair

        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=user.org_id,
            role=user.role,
        )

        logger.info(f"OAuth login: {user.email} via {user_info.provider}")

        redirect_url = state_data.get("redirect_url", _impl()._get_oauth_success_url())
        return self._redirect_with_tokens(redirect_url, tokens)

    async def _find_user_by_oauth(self, user_store, user_info: OAuthUserInfo):
        """Find user by OAuth provider ID (async version)."""
        # Look for user with matching OAuth link
        # This requires the user store to support OAuth lookups
        if hasattr(user_store, "get_user_by_oauth_async"):
            return await user_store.get_user_by_oauth_async(
                user_info.provider, user_info.provider_user_id
            )
        elif hasattr(user_store, "get_user_by_oauth"):
            # Fallback to sync method for non-async stores
            return user_store.get_user_by_oauth(user_info.provider, user_info.provider_user_id)
        return None

    async def _link_oauth_to_user(self, user_store, user_id: str, user_info: OAuthUserInfo) -> bool:
        """Link OAuth provider to existing user (async version)."""
        if hasattr(user_store, "link_oauth_provider_async"):
            return await user_store.link_oauth_provider_async(
                user_id=user_id,
                provider=user_info.provider,
                provider_user_id=user_info.provider_user_id,
                email=user_info.email,
            )
        elif hasattr(user_store, "link_oauth_provider"):
            return user_store.link_oauth_provider(
                user_id=user_id,
                provider=user_info.provider,
                provider_user_id=user_info.provider_user_id,
                email=user_info.email,
            )
        # Fallback: store in user metadata
        logger.warning("UserStore doesn't support OAuth linking, using fallback")
        return False

    async def _create_oauth_user(self, user_store, user_info: OAuthUserInfo):
        """Create a new user from OAuth info (async version)."""
        from aragora.billing.models import hash_password

        # Generate random password (user will use OAuth to login)
        random_password = secrets.token_urlsafe(32)
        password_hash, password_salt = hash_password(random_password)

        try:
            if hasattr(user_store, "create_user_async"):
                user = await user_store.create_user_async(
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

            logger.debug(f"OAuth user created: id={user.id}, email={user_info.email}")

            # Link OAuth provider
            await self._link_oauth_to_user(user_store, user.id, user_info)

            logger.info(f"Created OAuth user: {user_info.email} via {user_info.provider}")
            return user

        except ValueError as e:
            logger.error(f"Failed to create OAuth user: {e}")
            return None

    async def _handle_account_linking(
        self,
        user_store,
        user_id: str,
        user_info: OAuthUserInfo,
        state_data: dict,
    ) -> HandlerResult:
        """Handle linking OAuth account to existing user (async version)."""
        # Verify user exists
        if hasattr(user_store, "get_user_by_id_async"):
            user = await user_store.get_user_by_id_async(user_id)
        else:
            user = user_store.get_user_by_id(user_id)
        if not user:
            return self._redirect_with_error("User not found")

        # Check if OAuth is already linked to another account
        existing_user = await self._find_user_by_oauth(user_store, user_info)
        if existing_user and existing_user.id != user_id:
            return self._redirect_with_error(
                f"This {user_info.provider.title()} account is already linked to another user"
            )

        # Link OAuth
        success = await self._link_oauth_to_user(user_store, user_id, user_info)
        if not success:
            logger.warning(f"OAuth linking fallback for user {user_id}")

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

    def _redirect_with_tokens(self, redirect_url: str, tokens) -> HandlerResult:
        """Redirect to frontend with tokens in URL query parameters.

        Note: We use query params instead of URL fragments because fragments
        are stripped during HTTP redirects (by proxies like Cloudflare).
        The frontend callback page handles extracting tokens from query params.
        """
        params = urlencode(
            {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "token_type": "Bearer",
                "expires_in": tokens.expires_in,
            }
        )
        url = f"{redirect_url}?{params}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={url}"></head></html>'.encode(),
            headers={"Location": url, **self.OAUTH_NO_CACHE_HEADERS},
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
