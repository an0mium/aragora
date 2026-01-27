"""
OAuth Authentication Handlers.

Supports:
- Google OAuth 2.0 (primary)
- Extensible for future OAuth providers (GitHub, Microsoft, etc.)

Endpoints:
- GET /api/auth/oauth/google - Redirect to Google OAuth consent screen
- GET /api/auth/oauth/google/callback - Handle OAuth callback
- POST /api/auth/oauth/link - Link OAuth account to existing user
- DELETE /api/auth/oauth/unlink - Unlink OAuth provider from account
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import urlencode

from aragora.audit.unified import audit_action, audit_security

from .base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from .secure import SecureHandler
from .utils.rate_limit import RateLimiter, get_client_ip

# RBAC imports
from aragora.rbac import AuthorizationContext, check_permission
from aragora.rbac.defaults import get_role_permissions

logger = logging.getLogger(__name__)

# Rate limiter for OAuth endpoints (20 requests per minute - auth attempts should be limited)
_oauth_limiter = RateLimiter(requests_per_minute=20)


# =============================================================================
# Configuration (loaded lazily to support AWS Secrets Manager)
# =============================================================================


def _get_secret(name: str, default: str = "") -> str:
    """Get a secret from AWS Secrets Manager or environment."""
    try:
        from aragora.config.secrets import get_secret

        return get_secret(name, default) or default
    except ImportError:
        return os.environ.get(name, default)


def _is_production() -> bool:
    """Check if we're in production mode."""
    return os.environ.get("ARAGORA_ENV", "").lower() == "production"


def _get_google_client_id() -> str:
    return _get_secret("GOOGLE_OAUTH_CLIENT_ID", "")


def _get_google_client_secret() -> str:
    return _get_secret("GOOGLE_OAUTH_CLIENT_SECRET", "")


def _get_github_client_id() -> str:
    return _get_secret("GITHUB_OAUTH_CLIENT_ID", "")


def _get_github_client_secret() -> str:
    return _get_secret("GITHUB_OAUTH_CLIENT_SECRET", "")


def _get_microsoft_client_id() -> str:
    return _get_secret("MICROSOFT_OAUTH_CLIENT_ID", "")


def _get_microsoft_client_secret() -> str:
    return _get_secret("MICROSOFT_OAUTH_CLIENT_SECRET", "")


def _get_microsoft_tenant() -> str:
    """Get Microsoft tenant ID (default: 'common' for multi-tenant)."""
    return _get_secret("MICROSOFT_OAUTH_TENANT", "common")


def _get_apple_client_id() -> str:
    return _get_secret("APPLE_OAUTH_CLIENT_ID", "")


def _get_apple_team_id() -> str:
    return _get_secret("APPLE_TEAM_ID", "")


def _get_apple_key_id() -> str:
    return _get_secret("APPLE_KEY_ID", "")


def _get_apple_private_key() -> str:
    return _get_secret("APPLE_PRIVATE_KEY", "")


def _get_oidc_issuer() -> str:
    return _get_secret("OIDC_ISSUER", "")


def _get_oidc_client_id() -> str:
    return _get_secret("OIDC_CLIENT_ID", "")


def _get_oidc_client_secret() -> str:
    return _get_secret("OIDC_CLIENT_SECRET", "")


def _get_google_redirect_uri() -> str:
    val = _get_secret("GOOGLE_OAUTH_REDIRECT_URI", "")
    if val:
        return val
    if _is_production():
        return ""
    return "http://localhost:8080/api/auth/oauth/google/callback"


def _get_github_redirect_uri() -> str:
    val = _get_secret("GITHUB_OAUTH_REDIRECT_URI", "")
    if val:
        return val
    if _is_production():
        return ""
    return "http://localhost:8080/api/auth/oauth/github/callback"


def _get_microsoft_redirect_uri() -> str:
    val = _get_secret("MICROSOFT_OAUTH_REDIRECT_URI", "")
    if val:
        return val
    if _is_production():
        return ""
    return "http://localhost:8080/api/auth/oauth/microsoft/callback"


def _get_apple_redirect_uri() -> str:
    val = _get_secret("APPLE_OAUTH_REDIRECT_URI", "")
    if val:
        return val
    if _is_production():
        return ""
    return "http://localhost:8080/api/auth/oauth/apple/callback"


def _get_oidc_redirect_uri() -> str:
    val = _get_secret("OIDC_REDIRECT_URI", "")
    if val:
        return val
    if _is_production():
        return ""
    return "http://localhost:8080/api/auth/oauth/oidc/callback"


def _get_oauth_success_url() -> str:
    val = _get_secret("OAUTH_SUCCESS_URL", "")
    if val:
        return val
    if _is_production():
        return ""
    return "http://localhost:3000/auth/callback"


def _get_oauth_error_url() -> str:
    val = _get_secret("OAUTH_ERROR_URL", "")
    if val:
        return val
    if _is_production():
        return ""
    return "http://localhost:3000/auth/error"


def _get_allowed_redirect_hosts() -> frozenset:
    val = _get_secret("OAUTH_ALLOWED_REDIRECT_HOSTS", "")
    if not val:
        if _is_production():
            return frozenset()
        val = "localhost,127.0.0.1"
    return frozenset(host.strip().lower() for host in val.split(",") if host.strip())


# Legacy module-level variables (for backward compatibility, now call functions)
# These are kept for any code that imports them directly
_IS_PRODUCTION = _is_production()
GOOGLE_CLIENT_ID = _get_google_client_id()
GOOGLE_CLIENT_SECRET = _get_google_client_secret()
GITHUB_CLIENT_ID = _get_github_client_id()
GITHUB_CLIENT_SECRET = _get_github_client_secret()
GOOGLE_REDIRECT_URI = _get_google_redirect_uri()
GITHUB_REDIRECT_URI = _get_github_redirect_uri()
OAUTH_SUCCESS_URL = _get_oauth_success_url()
OAUTH_ERROR_URL = _get_oauth_error_url()
ALLOWED_OAUTH_REDIRECT_HOSTS = _get_allowed_redirect_hosts()


def validate_oauth_config() -> list[str]:
    """
    Validate OAuth configuration and return list of missing required vars.

    Call this at startup to catch configuration errors early.
    Returns empty list if configuration is valid, or list of missing var names.
    """
    if not _IS_PRODUCTION:
        return []  # No validation in dev mode

    missing = []

    # If Google OAuth is enabled (client ID set), check required vars
    if GOOGLE_CLIENT_ID:
        if not _get_google_client_secret():
            missing.append("GOOGLE_OAUTH_CLIENT_SECRET")
        if not _get_google_redirect_uri():
            missing.append("GOOGLE_OAUTH_REDIRECT_URI")
        if not _get_oauth_success_url():
            missing.append("_get_oauth_success_url()")
        if not _get_oauth_error_url():
            missing.append("_get_oauth_error_url()")
        if not ALLOWED_OAUTH_REDIRECT_HOSTS:
            missing.append("OAUTH_ALLOWED_REDIRECT_HOSTS")

    # If GitHub OAuth is enabled (client ID set), check required vars
    if GITHUB_CLIENT_ID:
        if not _get_github_client_secret():
            missing.append("GITHUB_OAUTH_CLIENT_SECRET")
        if not _get_github_redirect_uri():
            missing.append("GITHUB_OAUTH_REDIRECT_URI")
        # Shared URLs only need to be set once
        if not _get_oauth_success_url() and "_get_oauth_success_url()" not in missing:
            missing.append("_get_oauth_success_url()")
        if not _get_oauth_error_url() and "_get_oauth_error_url()" not in missing:
            missing.append("_get_oauth_error_url()")
        if not ALLOWED_OAUTH_REDIRECT_HOSTS and "OAUTH_ALLOWED_REDIRECT_HOSTS" not in missing:
            missing.append("OAUTH_ALLOWED_REDIRECT_HOSTS")

    return missing


# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# GitHub OAuth endpoints
GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USERINFO_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

# Microsoft OAuth endpoints (Azure AD v2.0)
# Note: {tenant} is replaced at runtime with the configured tenant
MICROSOFT_AUTH_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
MICROSOFT_USERINFO_URL = "https://graph.microsoft.com/v1.0/me"

# Apple OAuth endpoints
APPLE_AUTH_URL = "https://appleid.apple.com/auth/authorize"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"

# State management - uses Redis in production, falls back to in-memory
# Import from dedicated state store module
from aragora.server.oauth_state_store import (
    OAuthState,
    get_oauth_state_store,
)
from aragora.server.oauth_state_store import (
    generate_oauth_state as _generate_state,
)
from aragora.server.oauth_state_store import (
    validate_oauth_state as _validate_state_internal,
)


class _OAuthStatesView(MutableMapping[str, dict]):
    """Compatibility view over OAuth state storage."""

    def __init__(self, store) -> None:
        self._store = store

    @property
    def _states(self) -> dict:
        return self._store._memory_store._states  # type: ignore[attr-defined]

    def __getitem__(self, key: str) -> dict:
        value = self._states[key]
        if isinstance(value, OAuthState):
            return value.to_dict()
        if isinstance(value, dict):
            return value
        return {"value": value}

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, OAuthState):
            self._states[key] = value
            return
        if isinstance(value, dict):
            self._states[key] = OAuthState.from_dict(value)
            return
        self._states[key] = value

    def __delitem__(self, key: str) -> None:
        del self._states[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def values(self):
        return [self[k] for k in list(self._states.keys())]

    def items(self):
        return [(k, self[k]) for k in list(self._states.keys())]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._states:
            return self[key]
        return default


_state_store = get_oauth_state_store()
_OAUTH_STATES: Union[_OAuthStatesView, dict[str, Any]] = {}
try:
    _OAUTH_STATES = _OAuthStatesView(_state_store)
except AttributeError:
    pass  # Keep empty dict fallback

# Legacy constants for backward compatibility (actual values from oauth_state_store)
_STATE_TTL_SECONDS = 600  # 10 minutes
MAX_OAUTH_STATES = 10000  # Prevent memory exhaustion from rapid state generation


@dataclass
class OAuthUserInfo:
    """User info from OAuth provider."""

    provider: str
    provider_user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    email_verified: bool = False


def _get_param(query_params: dict, name: str, default: str = None) -> str:
    """
    Safely extract a query parameter value.

    Handler registry converts single-element lists to scalars, so we need to
    handle both list and string formats.

    Args:
        query_params: Dict of query parameters
        name: Parameter name to extract
        default: Default value if not found

    Returns:
        Parameter value as string, or default if not found
    """
    value = query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _validate_redirect_url(redirect_url: str) -> bool:
    """
    Validate that redirect URL is in the allowed hosts list and uses safe scheme.

    This prevents open redirect vulnerabilities where an attacker could
    craft an OAuth URL that redirects tokens to a malicious domain or uses
    dangerous URL schemes (javascript:, data:, etc.).

    Args:
        redirect_url: The URL to validate

    Returns:
        True if URL is allowed, False otherwise
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(redirect_url)

        # Security: Only allow http/https schemes to prevent javascript:/data:/etc attacks
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"oauth_redirect_blocked: scheme={parsed.scheme} not allowed")
            return False

        host = parsed.hostname
        if not host:
            return False

        # Normalize host for comparison
        host = host.lower()

        # Get allowed hosts at runtime from Secrets Manager
        allowed_hosts = _get_allowed_redirect_hosts()

        # Check against allowlist
        if host in allowed_hosts:
            return True

        # Check if it's a subdomain of allowed hosts
        for allowed in allowed_hosts:
            if host.endswith(f".{allowed}"):
                return True

        logger.warning(f"oauth_redirect_blocked: host={host} not in allowlist")
        return False
    except Exception as e:
        logger.warning(f"oauth_redirect_validation_error: {e}")
        return False


def _validate_state(state: str) -> Optional[dict[str, Any]]:
    """Validate and consume OAuth state token.

    Uses Redis in production for multi-instance support,
    falls back to in-memory storage in development.
    """
    return _validate_state_internal(state)


def _cleanup_expired_states() -> int:
    """Backward-compatible cleanup helper for in-memory states."""
    try:
        return _state_store._memory_store.cleanup_expired()  # type: ignore[attr-defined]
    except AttributeError:
        return 0


class OAuthHandler(SecureHandler):
    """Handler for OAuth authentication endpoints.

    Extends SecureHandler for JWT-based authentication, RBAC permission
    enforcement, and security audit logging.
    """

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
    ) -> Optional[HandlerResult]:
        """Route OAuth requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _oauth_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for OAuth endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if hasattr(handler, "command"):
            method = handler.command

        # Normalize path - support both /api/v1/ and /api/ prefixes
        normalized = path.replace("/api/v1/", "/api/")

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

        if normalized == "/api/auth/oauth/link" and method == "POST":
            return self._handle_link_account(handler)

        if normalized == "/api/auth/oauth/unlink" and method == "DELETE":
            return self._handle_unlink_account(handler)

        if normalized == "/api/auth/oauth/providers" and method == "GET":
            return self._handle_list_providers(handler)

        if normalized == "/api/user/oauth-providers" and method == "GET":
            return self._handle_get_user_providers(handler)

        return error_response("Method not allowed", 405)

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _check_permission(
        self, handler, permission_key: str, resource_id: str | None = None
    ) -> Optional[HandlerResult]:
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

    @handle_errors("Google OAuth start")
    @log_request("Google OAuth start")
    def _handle_google_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to Google OAuth consent screen."""
        google_client_id = _get_google_client_id()
        if not google_client_id:
            return error_response("Google OAuth not configured", 503)

        # Get optional redirect URL from query params
        oauth_success_url = _get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        # Security: Validate redirect URL against allowlist to prevent open redirects
        if not _validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL. Only approved domains are allowed.", 400)

        # Check if this is for account linking (user already authenticated)
        user_id = None
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if auth_ctx.is_authenticated:
            user_id = auth_ctx.user_id

        # Generate state for CSRF protection
        state = _generate_state(user_id=user_id, redirect_url=redirect_url)

        # Build authorization URL
        params = {
            "client_id": google_client_id,
            "redirect_uri": _get_google_redirect_uri(),
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

        # Return redirect response
        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={auth_url}"></head></html>'.encode(),
            headers={"Location": auth_url},
        )

    @handle_errors("Google OAuth callback")
    @log_request("Google OAuth callback")
    def _handle_google_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle Google OAuth callback with authorization code."""

        # Check for error from Google
        error = _get_param(query_params, "error")
        if error:
            error_desc = _get_param(query_params, "error_description", error)
            logger.warning(f"Google OAuth error: {error} - {error_desc}")
            # Audit failed OAuth attempt
            audit_security(
                event_type="oauth_failed",
                actor_id="unknown",
                resource_type="auth",
                provider="google",
                error=error,
                reason=error_desc,
            )
            return self._redirect_with_error(f"OAuth error: {error_desc}")

        # Validate state
        state = _get_param(query_params, "state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        logger.info(f"OAuth callback: validating state (len={len(state)}, prefix={state[:20]}...)")
        state_data = _validate_state(state)
        if state_data is None:
            logger.warning(f"OAuth callback: state validation failed for {state[:20]}...")
            return self._redirect_with_error("Invalid or expired state")
        logger.info(
            f"OAuth callback: state valid, redirect_url={state_data.get('redirect_url', 'NOT_SET')}"
        )

        # Get authorization code
        code = _get_param(query_params, "code")
        if not code:
            return self._redirect_with_error("Missing authorization code")

        # Exchange code for tokens
        try:
            logger.info("OAuth callback: exchanging code for tokens...")
            token_data = self._exchange_code_for_tokens(code)
            logger.info("OAuth callback: token exchange successful")
        except Exception as e:
            logger.error(f"Token exchange failed: {e}", exc_info=True)
            return self._redirect_with_error("Failed to exchange authorization code")

        access_token = token_data.get("access_token")
        if not access_token:
            return self._redirect_with_error("No access token received")

        # Get user info from Google
        try:
            logger.info("OAuth callback: fetching user info from Google...")
            user_info = self._get_google_user_info(access_token)
            logger.info(f"OAuth callback: got user info for {user_info.email}")
        except Exception as e:
            logger.error(f"Failed to get user info: {e}", exc_info=True)
            return self._redirect_with_error("Failed to get user info from Google")

        # Handle user creation/login
        user_store = self._get_user_store()
        if not user_store:
            logger.error("OAuth callback: user_store is None!")
            return self._redirect_with_error("User service unavailable")

        # Check if this is account linking
        linking_user_id = state_data.get("user_id")
        if linking_user_id:
            return self._handle_account_linking(user_store, linking_user_id, user_info, state_data)

        # Check if user exists by OAuth provider ID
        try:
            logger.info("OAuth callback: looking up user by OAuth ID...")
            user = self._find_user_by_oauth(user_store, user_info)
            logger.info(f"OAuth callback: find_user_by_oauth returned {'user' if user else 'None'}")
        except Exception as e:
            logger.error(f"OAuth callback: _find_user_by_oauth failed: {e}", exc_info=True)
            raise

        if not user:
            # Check if email already registered (without OAuth)
            try:
                logger.info(f"OAuth callback: looking up user by email {user_info.email}...")
                user = user_store.get_user_by_email(user_info.email)
                logger.info(
                    f"OAuth callback: get_user_by_email returned {'user' if user else 'None'}"
                )
            except Exception as e:
                logger.error(f"OAuth callback: get_user_by_email failed: {e}", exc_info=True)
                raise

            if user:
                # Link OAuth to existing account
                logger.info(f"OAuth callback: linking OAuth to existing user {user.id}")
                self._link_oauth_to_user(user_store, user.id, user_info)
            else:
                # Create new user with OAuth
                try:
                    logger.info(f"OAuth callback: creating new OAuth user for {user_info.email}...")
                    user = self._create_oauth_user(user_store, user_info)
                    logger.info(f"OAuth callback: created user {user.id if user else 'FAILED'}")
                except Exception as e:
                    logger.error(f"OAuth callback: _create_oauth_user failed: {e}", exc_info=True)
                    raise

        if not user:
            return self._redirect_with_error("Failed to create user account")

        # Update last login
        try:
            logger.info(f"OAuth callback: updating last login for user {user.id}...")
            user_store.update_user(user.id, last_login_at=time.time())
        except Exception as e:
            logger.error(f"OAuth callback: update_user failed: {e}", exc_info=True)
            # Non-fatal, continue

        # Create tokens
        try:
            logger.info(f"OAuth callback: creating token pair for user {user.id}...")
            from aragora.billing.jwt_auth import create_token_pair

            tokens = create_token_pair(
                user_id=user.id,
                email=user.email,
                org_id=user.org_id,
                role=user.role,
            )
            # Log token fingerprint for debugging (correlates with validation logs)
            import hashlib

            token_fingerprint = hashlib.sha256(tokens.access_token.encode()).hexdigest()[:8]
            logger.info(
                f"OAuth callback: token pair created successfully "
                f"(access_token fingerprint={token_fingerprint}, "
                f"user_id={user.id}, org_id={user.org_id})"
            )
        except Exception as e:
            logger.error(f"OAuth callback: create_token_pair failed: {e}", exc_info=True)
            raise

        logger.info(f"OAuth login successful: {user.email} via Google")

        # Audit successful OAuth login
        audit_action(
            user_id=user.id,
            action="oauth_login",
            resource_type="auth",
            resource_id=user.id,
            provider="google",
            email=user.email,
            success=True,
        )

        # Redirect to frontend with tokens
        redirect_url = state_data.get("redirect_url", _get_oauth_success_url())
        logger.info(f"OAuth callback: redirecting to {redirect_url}")
        return self._redirect_with_tokens(redirect_url, tokens)

    def _exchange_code_for_tokens(self, code: str) -> dict:
        """Exchange authorization code for access tokens."""
        import urllib.error
        import urllib.request

        data = urlencode(
            {
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": _get_google_client_secret(),
                "redirect_uri": _get_google_redirect_uri(),
                "grant_type": "authorization_code",
            }
        ).encode()

        req = urllib.request.Request(
            GOOGLE_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            try:
                return json.loads(response.read().decode())
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from Google token endpoint: {e}")
                raise ValueError(f"Invalid JSON response from Google: {e}") from e

    def _get_google_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user info from Google API."""
        import urllib.request

        req = urllib.request.Request(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            try:
                data = json.loads(response.read().decode())
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from Google userinfo endpoint: {e}")
                raise ValueError(f"Invalid JSON response from Google: {e}") from e

        return OAuthUserInfo(
            provider="google",
            provider_user_id=data["id"],
            email=data["email"],
            name=data.get("name", data["email"].split("@")[0]),
            picture=data.get("picture"),
            email_verified=data.get("verified_email", False),
        )

    # =========================================================================
    # GitHub OAuth Methods
    # =========================================================================

    @handle_errors("GitHub OAuth start")
    @log_request("GitHub OAuth start")
    def _handle_github_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to GitHub OAuth consent screen."""
        github_client_id = _get_github_client_id()
        if not github_client_id:
            return error_response("GitHub OAuth not configured", 503)

        # Get optional redirect URL from query params
        oauth_success_url = _get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        # Security: Validate redirect URL against allowlist
        if not _validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL. Only approved domains are allowed.", 400)

        # Check if this is for account linking (user already authenticated)
        user_id = None
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if auth_ctx.is_authenticated:
            user_id = auth_ctx.user_id

        # Generate state for CSRF protection
        state = _generate_state(user_id=user_id, redirect_url=redirect_url)

        # Build authorization URL
        params = {
            "client_id": github_client_id,
            "redirect_uri": _get_github_redirect_uri(),
            "scope": "read:user user:email",
            "state": state,
        }
        auth_url = f"{GITHUB_AUTH_URL}?{urlencode(params)}"

        # Return redirect response
        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={auth_url}"></head></html>'.encode(),
            headers={"Location": auth_url},
        )

    @handle_errors("GitHub OAuth callback")
    @log_request("GitHub OAuth callback")
    def _handle_github_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle GitHub OAuth callback with authorization code."""

        # Check for error from GitHub
        error = _get_param(query_params, "error")
        if error:
            error_desc = _get_param(query_params, "error_description", error)
            logger.warning(f"GitHub OAuth error: {error} - {error_desc}")
            return self._redirect_with_error(f"OAuth error: {error_desc}")

        # Validate state
        state = _get_param(query_params, "state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = _validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        # Get authorization code
        code = _get_param(query_params, "code")
        if not code:
            return self._redirect_with_error("Missing authorization code")

        # Exchange code for tokens
        try:
            token_data = self._exchange_github_code(code)
        except Exception as e:
            logger.error(f"GitHub token exchange failed: {e}")
            return self._redirect_with_error("Failed to exchange authorization code")

        access_token = token_data.get("access_token")
        if not access_token:
            error_msg = token_data.get(
                "error_description", token_data.get("error", "Unknown error")
            )
            logger.error(f"GitHub OAuth: No access token - {error_msg}")
            return self._redirect_with_error("No access token received from GitHub")

        # Get user info from GitHub
        try:
            user_info = self._get_github_user_info(access_token)
        except Exception as e:
            logger.error(f"Failed to get GitHub user info: {e}")
            return self._redirect_with_error("Failed to get user info from GitHub")

        # Handle user creation/login
        user_store = self._get_user_store()
        if not user_store:
            return self._redirect_with_error("User service unavailable")

        # Check if this is account linking
        linking_user_id = state_data.get("user_id")
        if linking_user_id:
            return self._handle_account_linking(user_store, linking_user_id, user_info, state_data)

        # Check if user exists by OAuth provider ID
        user = self._find_user_by_oauth(user_store, user_info)

        if not user:
            # Check if email already registered (without OAuth)
            user = user_store.get_user_by_email(user_info.email)
            if user:
                # Link OAuth to existing account
                self._link_oauth_to_user(user_store, user.id, user_info)
            else:
                # Create new user with OAuth
                user = self._create_oauth_user(user_store, user_info)

        if not user:
            return self._redirect_with_error("Failed to create user account")

        # Update last login
        user_store.update_user(user.id, last_login_at=time.time())

        # Create tokens
        from aragora.billing.jwt_auth import create_token_pair

        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=user.org_id,
            role=user.role,
        )

        logger.info(f"OAuth login: {user.email} via GitHub")

        # Redirect to frontend with tokens
        redirect_url = state_data.get("redirect_url", _get_oauth_success_url())
        return self._redirect_with_tokens(redirect_url, tokens)

    def _exchange_github_code(self, code: str) -> dict:
        """Exchange GitHub authorization code for access token."""
        import urllib.error
        import urllib.request

        data = urlencode(
            {
                "code": code,
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": _get_github_client_secret(),
                "redirect_uri": _get_github_redirect_uri(),
            }
        ).encode()

        req = urllib.request.Request(
            GITHUB_TOKEN_URL,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            try:
                return json.loads(response.read().decode())
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from GitHub token endpoint: {e}")
                raise ValueError(f"Invalid JSON response from GitHub: {e}") from e

    def _get_github_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user info from GitHub API."""
        import urllib.request

        # Get basic user info
        req = urllib.request.Request(
            GITHUB_USERINFO_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            try:
                user_data = json.loads(response.read().decode())
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from GitHub user endpoint: {e}")
                raise ValueError(f"Invalid JSON response from GitHub: {e}") from e

        # Get user's emails (need to find primary verified email)
        email = user_data.get("email")
        email_verified = False

        if not email:
            # Email not public, fetch from emails endpoint
            email_req = urllib.request.Request(
                GITHUB_EMAILS_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )

            with urllib.request.urlopen(email_req, timeout=10) as response:
                try:
                    emails = json.loads(response.read().decode())
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from GitHub emails endpoint: {e}")
                    raise ValueError(f"Invalid JSON from GitHub emails: {e}") from e

                # Find primary verified email
                for email_entry in emails:
                    if email_entry.get("primary") and email_entry.get("verified"):
                        email = email_entry.get("email")
                        email_verified = True
                        break

                # Fallback to any verified email
                if not email:
                    for email_entry in emails:
                        if email_entry.get("verified"):
                            email = email_entry.get("email")
                            email_verified = True
                            break

                # Last resort: any email
                if not email and emails:
                    email = emails[0].get("email")

        if not email:
            raise ValueError("Could not retrieve email from GitHub")

        return OAuthUserInfo(
            provider="github",
            provider_user_id=str(user_data["id"]),
            email=email,
            name=user_data.get("name") or user_data.get("login", email.split("@")[0]),
            picture=user_data.get("avatar_url"),
            email_verified=email_verified,
        )

    # =========================================================================
    # Microsoft OAuth
    # =========================================================================

    @handle_errors("Microsoft OAuth start")
    @log_request("Microsoft OAuth start")
    def _handle_microsoft_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to Microsoft OAuth consent screen."""
        microsoft_client_id = _get_microsoft_client_id()
        if not microsoft_client_id:
            return error_response("Microsoft OAuth not configured", 503)

        oauth_success_url = _get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        if not _validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL", 400)

        user_id = None
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if auth_ctx.is_authenticated:
            user_id = auth_ctx.user_id

        state = _generate_state(user_id=user_id, redirect_url=redirect_url)

        tenant = _get_microsoft_tenant()
        auth_url_base = MICROSOFT_AUTH_URL_TEMPLATE.format(tenant=tenant)

        params = {
            "client_id": microsoft_client_id,
            "redirect_uri": _get_microsoft_redirect_uri(),
            "response_type": "code",
            "scope": "openid email profile User.Read",
            "state": state,
            "response_mode": "query",
        }
        auth_url = f"{auth_url_base}?{urlencode(params)}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={auth_url}"></head></html>'.encode(),
            headers={"Location": auth_url},
        )

    @handle_errors("Microsoft OAuth callback")
    @log_request("Microsoft OAuth callback")
    def _handle_microsoft_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle Microsoft OAuth callback."""
        error = _get_param(query_params, "error")
        if error:
            error_desc = _get_param(query_params, "error_description", error)
            logger.warning(f"Microsoft OAuth error: {error} - {error_desc}")
            return self._redirect_with_error(f"OAuth error: {error_desc}")

        state = _get_param(query_params, "state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = _validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        code = _get_param(query_params, "code")
        if not code:
            return self._redirect_with_error("Missing authorization code")

        try:
            token_data = self._exchange_microsoft_code(code)
        except Exception as e:
            logger.error(f"Microsoft token exchange failed: {e}")
            return self._redirect_with_error("Failed to exchange authorization code")

        access_token = token_data.get("access_token")
        if not access_token:
            return self._redirect_with_error("No access token received")

        try:
            user_info = self._get_microsoft_user_info(access_token)
        except Exception as e:
            logger.error(f"Failed to get Microsoft user info: {e}")
            return self._redirect_with_error("Failed to get user info")

        return self._complete_oauth_flow(user_info, state_data)

    def _exchange_microsoft_code(self, code: str) -> dict:
        """Exchange Microsoft authorization code for access token."""
        import urllib.request

        tenant = _get_microsoft_tenant()
        token_url = MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant=tenant)

        data = urlencode(
            {
                "code": code,
                "client_id": _get_microsoft_client_id(),
                "client_secret": _get_microsoft_client_secret(),
                "redirect_uri": _get_microsoft_redirect_uri(),
                "grant_type": "authorization_code",
            }
        ).encode()

        req = urllib.request.Request(
            token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    def _get_microsoft_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user info from Microsoft Graph API."""
        import urllib.request

        req = urllib.request.Request(
            MICROSOFT_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            user_data = json.loads(response.read().decode())

        email = user_data.get("mail") or user_data.get("userPrincipalName", "")
        if not email or "@" not in email:
            raise ValueError("Could not retrieve email from Microsoft")

        return OAuthUserInfo(
            provider="microsoft",
            provider_user_id=user_data["id"],
            email=email,
            name=user_data.get("displayName", email.split("@")[0]),
            picture=None,  # Microsoft Graph requires separate call for photo
            email_verified=True,  # Microsoft validates emails
        )

    # =========================================================================
    # Apple OAuth (Sign in with Apple)
    # =========================================================================

    @handle_errors("Apple OAuth start")
    @log_request("Apple OAuth start")
    def _handle_apple_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to Apple OAuth consent screen."""
        apple_client_id = _get_apple_client_id()
        if not apple_client_id:
            return error_response("Apple OAuth not configured", 503)

        oauth_success_url = _get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        if not _validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL", 400)

        user_id = None
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if auth_ctx.is_authenticated:
            user_id = auth_ctx.user_id

        state = _generate_state(user_id=user_id, redirect_url=redirect_url)

        params = {
            "client_id": apple_client_id,
            "redirect_uri": _get_apple_redirect_uri(),
            "response_type": "code id_token",
            "scope": "name email",
            "state": state,
            "response_mode": "form_post",  # Apple requires form_post
        }
        auth_url = f"{APPLE_AUTH_URL}?{urlencode(params)}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={auth_url}"></head></html>'.encode(),
            headers={"Location": auth_url},
        )

    @handle_errors("Apple OAuth callback")
    @log_request("Apple OAuth callback")
    def _handle_apple_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle Apple OAuth callback (POST with form data)."""
        # Apple uses form_post, so we need to read POST body
        body_data: dict[str, str] = {}
        if hasattr(handler, "request") and handler.request.body:
            from urllib.parse import parse_qs

            parsed = parse_qs(handler.request.body.decode())
            body_data = {k: v[0] if v else "" for k, v in parsed.items()}

        # Merge with query params (for GET fallback)
        all_params = {**query_params, **body_data}

        error = all_params.get("error")
        if error:
            return self._redirect_with_error(f"Apple OAuth error: {error}")

        state = all_params.get("state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = _validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        code = all_params.get("code")
        id_token = all_params.get("id_token")

        if not code and not id_token:
            return self._redirect_with_error("Missing authorization code or id_token")

        # Apple provides user info only on first authorization
        user_data_str = all_params.get("user", "{}")
        try:
            user_data = json.loads(user_data_str) if user_data_str else {}
        except json.JSONDecodeError:
            user_data = {}

        try:
            if code:
                token_data = self._exchange_apple_code(code)
                id_token = token_data.get("id_token", id_token)

            user_info = self._parse_apple_id_token(id_token, user_data)
        except Exception as e:
            logger.error(f"Apple OAuth processing failed: {e}")
            return self._redirect_with_error("Failed to process Apple sign-in")

        return self._complete_oauth_flow(user_info, state_data)

    def _exchange_apple_code(self, code: str) -> dict:
        """Exchange Apple authorization code for tokens."""
        import urllib.request

        client_secret = self._generate_apple_client_secret()

        data = urlencode(
            {
                "code": code,
                "client_id": _get_apple_client_id(),
                "client_secret": client_secret,
                "redirect_uri": _get_apple_redirect_uri(),
                "grant_type": "authorization_code",
            }
        ).encode()

        req = urllib.request.Request(
            APPLE_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    def _generate_apple_client_secret(self) -> str:
        """Generate Apple client secret JWT."""
        try:
            import jwt
        except ImportError:
            raise ValueError("PyJWT required for Apple OAuth. Install with: pip install PyJWT")

        team_id = _get_apple_team_id()
        key_id = _get_apple_key_id()
        private_key = _get_apple_private_key()
        client_id = _get_apple_client_id()

        if not all([team_id, key_id, private_key, client_id]):
            raise ValueError("Apple OAuth not fully configured")

        now = int(time.time())
        payload = {
            "iss": team_id,
            "iat": now,
            "exp": now + 86400 * 180,  # 180 days max
            "aud": "https://appleid.apple.com",
            "sub": client_id,
        }

        return jwt.encode(payload, private_key, algorithm="ES256", headers={"kid": key_id})

    def _parse_apple_id_token(self, id_token: str, user_data: dict) -> OAuthUserInfo:
        """Parse Apple ID token to extract user info."""
        import base64

        # Decode JWT payload (Apple signs it, but we trust it from their endpoint)
        parts = id_token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid Apple ID token format")

        # Decode payload with padding
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        email = payload.get("email", "")
        email_verified = payload.get("email_verified", False)
        sub = payload.get("sub", "")

        if not email:
            raise ValueError("No email in Apple ID token")

        # Apple only sends name on first auth, stored in user_data
        name_data = user_data.get("name", {})
        name = ""
        if name_data:
            first = name_data.get("firstName", "")
            last = name_data.get("lastName", "")
            name = f"{first} {last}".strip()
        if not name:
            name = email.split("@")[0]

        return OAuthUserInfo(
            provider="apple",
            provider_user_id=sub,
            email=email,
            name=name,
            picture=None,  # Apple doesn't provide profile pictures
            email_verified=email_verified == "true" or email_verified is True,
        )

    # =========================================================================
    # Generic OIDC Provider
    # =========================================================================

    @handle_errors("OIDC OAuth start")
    @log_request("OIDC OAuth start")
    def _handle_oidc_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to generic OIDC provider."""
        oidc_issuer = _get_oidc_issuer()
        oidc_client_id = _get_oidc_client_id()

        if not oidc_issuer or not oidc_client_id:
            return error_response("OIDC provider not configured", 503)

        oauth_success_url = _get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        if not _validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL", 400)

        user_id = None
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if auth_ctx.is_authenticated:
            user_id = auth_ctx.user_id

        state = _generate_state(user_id=user_id, redirect_url=redirect_url)

        # Discover OIDC endpoints
        discovery = self._get_oidc_discovery(oidc_issuer)
        auth_endpoint = discovery.get("authorization_endpoint")

        if not auth_endpoint:
            return error_response("OIDC discovery failed", 503)

        params = {
            "client_id": oidc_client_id,
            "redirect_uri": _get_oidc_redirect_uri(),
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
        }
        auth_url = f"{auth_endpoint}?{urlencode(params)}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={auth_url}"></head></html>'.encode(),
            headers={"Location": auth_url},
        )

    @handle_errors("OIDC OAuth callback")
    @log_request("OIDC OAuth callback")
    def _handle_oidc_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle generic OIDC callback."""
        error = _get_param(query_params, "error")
        if error:
            error_desc = _get_param(query_params, "error_description", error)
            return self._redirect_with_error(f"OIDC error: {error_desc}")

        state = _get_param(query_params, "state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = _validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        code = _get_param(query_params, "code")
        if not code:
            return self._redirect_with_error("Missing authorization code")

        oidc_issuer = _get_oidc_issuer()
        discovery = self._get_oidc_discovery(oidc_issuer)

        try:
            token_data = self._exchange_oidc_code(code, discovery)
        except Exception as e:
            logger.error(f"OIDC token exchange failed: {e}")
            return self._redirect_with_error("Failed to exchange authorization code")

        access_token = token_data.get("access_token")
        id_token = token_data.get("id_token")

        try:
            user_info = self._get_oidc_user_info(access_token, id_token, discovery)
        except Exception as e:
            logger.error(f"Failed to get OIDC user info: {e}")
            return self._redirect_with_error("Failed to get user info")

        return self._complete_oauth_flow(user_info, state_data)

    def _get_oidc_discovery(self, issuer: str) -> dict:
        """Fetch OIDC discovery document."""
        import urllib.request

        discovery_url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"

        try:
            req = urllib.request.Request(discovery_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            logger.error(f"OIDC discovery failed: {e}")
            return {}

    def _exchange_oidc_code(self, code: str, discovery: dict) -> dict:
        """Exchange OIDC authorization code for tokens."""
        import urllib.request

        token_endpoint = discovery.get("token_endpoint")
        if not token_endpoint:
            raise ValueError("No token endpoint in OIDC discovery")

        data = urlencode(
            {
                "code": code,
                "client_id": _get_oidc_client_id(),
                "client_secret": _get_oidc_client_secret(),
                "redirect_uri": _get_oidc_redirect_uri(),
                "grant_type": "authorization_code",
            }
        ).encode()

        req = urllib.request.Request(
            token_endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    def _get_oidc_user_info(
        self, access_token: str, id_token: str, discovery: dict
    ) -> OAuthUserInfo:
        """Get user info from OIDC userinfo endpoint or id_token."""
        import base64
        import urllib.request

        userinfo_endpoint = discovery.get("userinfo_endpoint")
        user_data = {}

        # Try userinfo endpoint first
        if userinfo_endpoint and access_token:
            try:
                req = urllib.request.Request(
                    userinfo_endpoint,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    user_data = json.loads(response.read().decode())
            except Exception as e:
                logger.warning(f"OIDC userinfo failed, falling back to id_token: {e}")

        # Fall back to id_token claims
        if not user_data and id_token:
            parts = id_token.split(".")
            if len(parts) == 3:
                payload_b64 = parts[1]
                padding = 4 - len(payload_b64) % 4
                if padding != 4:
                    payload_b64 += "=" * padding
                user_data = json.loads(base64.urlsafe_b64decode(payload_b64))

        email = user_data.get("email", "")
        if not email:
            raise ValueError("No email in OIDC response")

        sub = user_data.get("sub", "")
        if not sub:
            raise ValueError("No subject in OIDC response")

        return OAuthUserInfo(
            provider="oidc",
            provider_user_id=sub,
            email=email,
            name=user_data.get("name", email.split("@")[0]),
            picture=user_data.get("picture"),
            email_verified=user_data.get("email_verified", False),
        )

    # =========================================================================
    # Common OAuth Flow Completion
    # =========================================================================

    def _complete_oauth_flow(self, user_info: OAuthUserInfo, state_data: dict) -> HandlerResult:
        """Complete OAuth flow - create/login user and redirect with tokens."""
        user_store = self._get_user_store()
        if not user_store:
            return self._redirect_with_error("User service unavailable")

        # Check if this is account linking
        linking_user_id = state_data.get("user_id")
        if linking_user_id:
            return self._handle_account_linking(user_store, linking_user_id, user_info, state_data)

        # Check if user exists by OAuth provider ID
        user = self._find_user_by_oauth(user_store, user_info)

        if not user:
            # Check if email already registered
            user = user_store.get_user_by_email(user_info.email)
            if user:
                self._link_oauth_to_user(user_store, user.id, user_info)
            else:
                user = self._create_oauth_user(user_store, user_info)

        if not user:
            return self._redirect_with_error("Failed to create user account")

        # Update last login
        user_store.update_user(user.id, last_login_at=time.time())

        # Create tokens
        from aragora.billing.jwt_auth import create_token_pair

        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=user.org_id,
            role=user.role,
        )

        logger.info(f"OAuth login: {user.email} via {user_info.provider}")

        redirect_url = state_data.get("redirect_url", _get_oauth_success_url())
        return self._redirect_with_tokens(redirect_url, tokens)

    def _find_user_by_oauth(self, user_store, user_info: OAuthUserInfo):
        """Find user by OAuth provider ID."""
        # Look for user with matching OAuth link
        # This requires the user store to support OAuth lookups
        if hasattr(user_store, "get_user_by_oauth"):
            return user_store.get_user_by_oauth(user_info.provider, user_info.provider_user_id)
        return None

    def _link_oauth_to_user(self, user_store, user_id: str, user_info: OAuthUserInfo) -> bool:
        """Link OAuth provider to existing user."""
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

    def _create_oauth_user(self, user_store, user_info: OAuthUserInfo):
        """Create a new user from OAuth info."""
        from aragora.billing.models import hash_password

        # Generate random password (user will use OAuth to login)
        random_password = secrets.token_urlsafe(32)
        password_hash, password_salt = hash_password(random_password)

        try:
            user = user_store.create_user(
                email=user_info.email,
                password_hash=password_hash,
                password_salt=password_salt,
                name=user_info.name,
            )

            logger.debug(f"OAuth user created: id={user.id}, email={user_info.email}")

            # Link OAuth provider
            self._link_oauth_to_user(user_store, user.id, user_info)

            logger.info(f"Created OAuth user: {user_info.email} via {user_info.provider}")
            return user

        except ValueError as e:
            logger.error(f"Failed to create OAuth user: {e}")
            return None

    def _handle_account_linking(
        self,
        user_store,
        user_id: str,
        user_info: OAuthUserInfo,
        state_data: dict,
    ) -> HandlerResult:
        """Handle linking OAuth account to existing user."""
        # Verify user exists
        user = user_store.get_user_by_id(user_id)
        if not user:
            return self._redirect_with_error("User not found")

        # Check if OAuth is already linked to another account
        existing_user = self._find_user_by_oauth(user_store, user_info)
        if existing_user and existing_user.id != user_id:
            return self._redirect_with_error(
                f"This {user_info.provider.title()} account is already linked to another user"
            )

        # Link OAuth
        success = self._link_oauth_to_user(user_store, user_id, user_info)
        if not success:
            logger.warning(f"OAuth linking fallback for user {user_id}")

        redirect_url = state_data.get("redirect_url", _get_oauth_success_url())
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

        url = f"{_get_oauth_error_url()}?error={quote(error)}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={url}"></head></html>'.encode(),
            headers={"Location": url, **self.OAUTH_NO_CACHE_HEADERS},
        )

    @handle_errors("list OAuth providers")
    def _handle_list_providers(self, handler) -> HandlerResult:
        """List configured OAuth providers."""
        providers = []

        if _get_google_client_id():
            providers.append(
                {
                    "id": "google",
                    "name": "Google",
                    "enabled": True,
                    "auth_url": "/api/v1/auth/oauth/google",
                }
            )

        if _get_github_client_id():
            providers.append(
                {
                    "id": "github",
                    "name": "GitHub",
                    "enabled": True,
                    "auth_url": "/api/v1/auth/oauth/github",
                }
            )

        if _get_microsoft_client_id():
            providers.append(
                {
                    "id": "microsoft",
                    "name": "Microsoft",
                    "enabled": True,
                    "auth_url": "/api/v1/auth/oauth/microsoft",
                }
            )

        if _get_apple_client_id():
            providers.append(
                {
                    "id": "apple",
                    "name": "Apple",
                    "enabled": True,
                    "auth_url": "/api/v1/auth/oauth/apple",
                }
            )

        if _get_oidc_issuer() and _get_oidc_client_id():
            providers.append(
                {
                    "id": "oidc",
                    "name": "SSO",
                    "enabled": True,
                    "auth_url": "/api/v1/auth/oauth/oidc",
                }
            )

        return json_response({"providers": providers})

    @handle_errors("get user OAuth providers")
    def _handle_get_user_providers(self, handler) -> HandlerResult:
        """Get OAuth providers linked to the current user."""
        # RBAC check: authentication.read permission required
        if error := self._check_permission(handler, "authentication.read"):
            return error

        from aragora.billing.jwt_auth import extract_user_from_request

        # Get current user (already verified by _check_permission)
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        # Get linked providers for this user
        providers = []
        if hasattr(user_store, "get_oauth_providers"):
            providers = user_store.get_oauth_providers(auth_ctx.user_id)
        elif hasattr(user_store, "_oauth_repo"):
            providers = user_store._oauth_repo.get_providers_for_user(auth_ctx.user_id)

        return json_response({"providers": providers})

    @handle_errors("link OAuth account")
    def _handle_link_account(self, handler) -> HandlerResult:
        """Link OAuth account to current user (initiated from settings)."""
        # RBAC check: authentication.update permission required
        if error := self._check_permission(handler, "authentication.update"):
            return error

        from aragora.billing.jwt_auth import extract_user_from_request

        # Get current user (already verified by _check_permission)
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        provider = body.get("provider", "").lower()
        if provider not in ["google", "github", "microsoft", "apple", "oidc"]:
            return error_response("Unsupported provider", 400)

        # Return the auth URL for the provider
        redirect_url = body.get("redirect_url", _get_oauth_success_url())

        # Validate redirect URL against allowlist (same as start flow)
        if not _validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL. Only approved domains are allowed.", 400)

        state = _generate_state(user_id=auth_ctx.user_id, redirect_url=redirect_url)

        if provider == "google":
            if not GOOGLE_CLIENT_ID:
                return error_response("Google OAuth not configured", 503)
            params = {
                "client_id": GOOGLE_CLIENT_ID,
                "redirect_uri": _get_google_redirect_uri(),
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
                "access_type": "offline",
                "prompt": "consent",
            }
            auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
            return json_response({"auth_url": auth_url})

        if provider == "github":
            if not GITHUB_CLIENT_ID:
                return error_response("GitHub OAuth not configured", 503)
            params = {
                "client_id": GITHUB_CLIENT_ID,
                "redirect_uri": _get_github_redirect_uri(),
                "scope": "read:user user:email",
                "state": state,
            }
            auth_url = f"{GITHUB_AUTH_URL}?{urlencode(params)}"
            return json_response({"auth_url": auth_url})

        return error_response("Provider not implemented", 501)

    @handle_errors("unlink OAuth account")
    def _handle_unlink_account(self, handler) -> HandlerResult:
        """Unlink OAuth provider from current user."""
        # RBAC check: authentication.update permission required
        if error := self._check_permission(handler, "authentication.update"):
            return error

        from aragora.billing.jwt_auth import extract_user_from_request

        # Get current user (already verified by _check_permission)
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        provider = body.get("provider", "").lower()
        if provider not in ["google", "github", "microsoft", "apple", "oidc"]:
            return error_response("Unsupported provider", 400)

        # Get user
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        # Ensure user has a password set (can't unlink all auth methods)
        if not user.password_hash:
            return error_response(
                "Cannot unlink OAuth - no password set. Set a password first.", 400
            )

        # Unlink provider
        if hasattr(user_store, "unlink_oauth_provider"):
            success = user_store.unlink_oauth_provider(auth_ctx.user_id, provider)
            if not success:
                return error_response("Failed to unlink provider", 500)
        else:
            logger.warning("UserStore doesn't support OAuth unlinking")

        logger.info(f"Unlinked {provider} from user {auth_ctx.user_id}")

        # Audit OAuth unlink
        audit_action(
            user_id=auth_ctx.user_id,
            action="oauth_unlink",
            resource_type="auth",
            resource_id=auth_ctx.user_id,
            provider=provider,
        )

        return json_response({"message": f"Unlinked {provider} successfully"})


__all__ = ["OAuthHandler", "validate_oauth_config"]
