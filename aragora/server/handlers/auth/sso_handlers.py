"""
HTTP API Handlers for SSO Authentication.

Provides REST APIs for SSO/OIDC authentication flow:
- Initiate SSO login
- Handle OAuth/OIDC callback
- Refresh tokens
- Logout

Endpoints:
- GET /api/v1/auth/sso/login - Get SSO authorization URL
- GET /api/v1/auth/sso/callback - Handle OAuth callback
- POST /api/v1/auth/sso/refresh - Refresh access token
- POST /api/v1/auth/sso/logout - Logout from SSO
- GET /api/v1/auth/sso/providers - List available providers
- GET /api/v1/auth/sso/config - Get provider configuration
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from aragora.config import resolve_db_path
from aragora.server.errors import safe_error_message
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    success_response,
)
from aragora.rbac.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import auth_rate_limit
from aragora.server.oauth_state_store import (
    OAUTH_STATE_TTL_SECONDS,
    get_oauth_state_store,
)

logger = logging.getLogger(__name__)

# Thread-safe SSO provider instances
_sso_providers: dict[str, Any] = {}
_sso_providers_lock = threading.Lock()

# In-memory auth sessions (used alongside the state store for session tracking)
_auth_sessions: dict[str, dict[str, Any]] = {}

# OAuth state store singleton (supports Redis/SQLite/JWT for multi-instance deployments)
_sso_state_store = None
_sso_state_store_lock = threading.Lock()

# Session TTL (use OAuth state store TTL for consistency)
AUTH_SESSION_TTL = OAUTH_STATE_TTL_SECONDS


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions from the in-memory auth sessions dict."""
    now = time.time()
    expired = [
        key
        for key, session in _auth_sessions.items()
        if now - session.get("created_at", 0) > AUTH_SESSION_TTL
    ]
    for key in expired:
        del _auth_sessions[key]


def _get_sso_state_store():
    """Get or create the SSO state store (thread-safe singleton)."""
    global _sso_state_store
    if _sso_state_store is None:
        with _sso_state_store_lock:
            if _sso_state_store is None:
                _sso_state_store = get_oauth_state_store(
                    sqlite_path=resolve_db_path("aragora_sso_state.db"),
                    use_sqlite=True,
                    use_jwt=True,
                )
    return _sso_state_store


def _get_sso_provider(provider_type: str = "oidc"):
    """Get or create SSO provider for type."""
    with _sso_providers_lock:
        if provider_type in _sso_providers:
            return _sso_providers[provider_type]

        try:
            if provider_type == "oidc":
                from aragora.auth.oidc import OIDCConfig, OIDCProvider
                from aragora.auth.sso import SSOProviderType

                # Get configuration from environment
                config = OIDCConfig(
                    provider_type=SSOProviderType.OIDC,
                    client_id=os.environ.get("OIDC_CLIENT_ID", ""),
                    client_secret=os.environ.get("OIDC_CLIENT_SECRET", ""),
                    issuer_url=os.environ.get("OIDC_ISSUER_URL", ""),
                    callback_url=os.environ.get(
                        "OIDC_CALLBACK_URL",
                        "http://localhost:8080/api/v1/auth/sso/callback",
                    ),
                    scopes=os.environ.get("OIDC_SCOPES", "openid,email,profile").split(","),
                )

                if config.client_id and config.issuer_url:
                    provider = OIDCProvider(config)
                    _sso_providers[provider_type] = provider
                    return provider

            elif provider_type == "google":
                from aragora.auth.oidc import OIDCConfig, OIDCProvider
                from aragora.auth.sso import SSOProviderType

                config = OIDCConfig(
                    provider_type=SSOProviderType.GOOGLE,
                    client_id=os.environ.get("GOOGLE_CLIENT_ID", ""),
                    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
                    issuer_url="https://accounts.google.com",
                    callback_url=os.environ.get(
                        "GOOGLE_CALLBACK_URL",
                        "http://localhost:8080/api/v1/auth/sso/callback",
                    ),
                    scopes=["openid", "email", "profile"],
                )

                if config.client_id:
                    provider = OIDCProvider(config)
                    _sso_providers[provider_type] = provider
                    return provider

            elif provider_type == "github":
                from aragora.auth.oidc import OIDCConfig, OIDCProvider
                from aragora.auth.sso import SSOProviderType

                config = OIDCConfig(
                    client_id=os.environ.get("GITHUB_CLIENT_ID", ""),
                    client_secret=os.environ.get("GITHUB_CLIENT_SECRET", ""),
                    authorization_endpoint="https://github.com/login/oauth/authorize",
                    token_endpoint="https://github.com/login/oauth/access_token",
                    userinfo_endpoint="https://api.github.com/user",
                    callback_url=os.environ.get(
                        "GITHUB_CALLBACK_URL",
                        "http://localhost:8080/api/v1/auth/sso/callback",
                    ),
                    scopes=["user:email", "read:user"],
                    use_pkce=False,  # GitHub doesn't support PKCE
                    provider_type=SSOProviderType.GITHUB,
                )

                if config.client_id:
                    provider = OIDCProvider(config)
                    _sso_providers[provider_type] = provider
                    return provider

            return None

        except Exception as e:
            logger.warning(f"Failed to initialize SSO provider {provider_type}: {e}")
            return None


# =============================================================================
# SSO Login Flow
# =============================================================================


# NOTE: SSO login is a public endpoint - users must be able to initiate SSO
# authentication before they have a token. RBAC protection via middleware.
@auth_rate_limit(
    requests_per_minute=20,
    limiter_name="auth_sso_login",
    endpoint_name="SSO login initiation",
)
async def handle_sso_login(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Initiate SSO login flow.

    GET /api/v1/auth/sso/login
    Query params:
        provider: str (optional, default "oidc") - Provider type
        redirect_url: str (optional) - Where to redirect after auth
    """
    try:
        provider_type = data.get("provider", "oidc")
        redirect_url = data.get("redirect_url", "/")

        provider = _get_sso_provider(provider_type)
        if not provider:
            return error_response(
                f"SSO provider '{provider_type}' not configured",
                status=503,
            )

        # Generate state token using OAuth state store (supports Redis/SQLite/JWT)
        state_store = _get_sso_state_store()
        state = state_store.generate(
            redirect_url=redirect_url,
            metadata={"provider_type": provider_type},
        )

        # Store session in-memory for quick lookup (kept alongside state store)
        _cleanup_expired_sessions()
        _auth_sessions[state] = {
            "created_at": time.time(),
            "redirect_url": redirect_url,
            "provider_type": provider_type,
        }

        # Get authorization URL
        auth_url = await provider.get_authorization_url(state=state)

        return success_response(
            {
                "authorization_url": auth_url,
                "state": state,
                "provider": provider_type,
                "expires_in": AUTH_SESSION_TTL,
            }
        )

    except Exception as e:
        logger.exception("SSO login initiation failed")
        return error_response(safe_error_message(e, "SSO login"), status=500)


# NOTE: SSO callback is a public endpoint - IdP redirects here before user
# has our token. RBAC protection via middleware.
@auth_rate_limit(
    requests_per_minute=20,
    limiter_name="auth_sso_callback",
    endpoint_name="SSO callback",
)
async def handle_sso_callback(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Handle SSO callback from IdP.

    GET /api/v1/auth/sso/callback
    Query params:
        code: str - Authorization code
        state: str - State parameter for CSRF
        error: str (optional) - Error from IdP
        error_description: str (optional)
    """
    try:
        # Check for IdP error
        if data.get("error"):
            error_msg = data.get("error_description", data["error"])
            logger.warning(f"SSO callback error from IdP: {error_msg}")
            return error_response(f"SSO error: {error_msg}", status=401)

        code = data.get("code")
        state = data.get("state")

        if not code:
            return error_response("No authorization code provided", status=400)
        if not state:
            return error_response("No state parameter provided", status=400)

        # Validate state using OAuth state store (atomic get-and-delete)
        # NOTE: State validation is timing-safe:
        # - JWTOAuthStateStore uses hmac.compare_digest() for signature verification
        # - InMemoryOAuthStateStore uses dictionary lookup (hash-based, constant-time)
        # - SQLiteOAuthStateStore uses SQL index lookup (not timing-vulnerable)
        # - RedisOAuthStateStore uses Redis key lookup (not timing-vulnerable)
        state_store = _get_sso_state_store()
        oauth_state = state_store.validate_and_consume(state)
        if not oauth_state:
            _cleanup_expired_sessions()
            session = _auth_sessions.pop(state, None)
            if not session:
                return error_response("Invalid or expired state", status=401)
            provider_type = session.get("provider_type", "oidc")
            redirect_url = session.get("redirect_url", "/")
        else:
            # Extract session data from state
            provider_type = (oauth_state.metadata or {}).get("provider_type", "oidc")
            redirect_url = oauth_state.redirect_url or "/"

        provider = _get_sso_provider(provider_type)
        if not provider:
            return error_response("SSO provider not available", status=503)

        # Authenticate with the provider
        sso_user = await provider.authenticate(code=code, state=state)

        # Create or update user in our system
        from aragora.billing.jwt_auth import create_access_token

        # Generate JWT token for our system
        access_token = create_access_token(
            user_id=sso_user.id,
            email=sso_user.email,
        )

        return success_response(
            {
                "access_token": access_token,
                "token_type": "bearer",
                "user": sso_user.to_dict(),
                "redirect_url": redirect_url,
                "sso_access_token": sso_user.access_token,  # For API calls to IdP
                "expires_at": sso_user.token_expires_at,
            }
        )

    except Exception as e:
        logger.exception("SSO callback failed")
        return error_response(safe_error_message(e, "SSO authentication"), status=401)


@require_permission("auth:manage_sso")
@auth_rate_limit(
    requests_per_minute=20,
    limiter_name="auth_sso_refresh",
    endpoint_name="SSO token refresh",
)
async def handle_sso_refresh(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Refresh SSO access token.

    POST /api/v1/auth/sso/refresh
    Body: {
        provider: str,
        refresh_token: str
    }
    """
    try:
        provider_type = data.get("provider", "oidc")
        refresh_token = data.get("refresh_token")

        if not refresh_token:
            return error_response("refresh_token is required", status=400)

        provider = _get_sso_provider(provider_type)
        if not provider:
            return error_response("SSO provider not available", status=503)

        # Create minimal user for refresh
        from aragora.auth.sso import SSOUser

        temp_user = SSOUser(
            id=user_id,
            email="",
            refresh_token=refresh_token,
        )

        refreshed_user = await provider.refresh_token(temp_user)

        if not refreshed_user:
            return error_response("Token refresh failed", status=401)

        return success_response(
            {
                "access_token": refreshed_user.access_token,
                "refresh_token": refreshed_user.refresh_token,
                "expires_at": refreshed_user.token_expires_at,
            }
        )

    except Exception as e:
        logger.exception("SSO refresh failed")
        return error_response(safe_error_message(e, "token refresh"), status=401)


@require_permission("auth:manage_sso")
@auth_rate_limit(
    requests_per_minute=20,
    limiter_name="auth_sso_logout",
    endpoint_name="SSO logout",
)
async def handle_sso_logout(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Logout from SSO.

    POST /api/v1/auth/sso/logout
    Body: {
        provider: str,
        id_token: str (optional) - For IdP logout
    }
    """
    try:
        provider_type = data.get("provider", "oidc")
        id_token = data.get("id_token")

        provider = _get_sso_provider(provider_type)

        logout_url = None
        if provider and id_token:
            from aragora.auth.sso import SSOUser

            temp_user = SSOUser(
                id=user_id,
                email="",
                id_token=id_token,
            )
            logout_url = await provider.logout(temp_user)

        return success_response(
            {
                "logged_out": True,
                "logout_url": logout_url,  # If provided, redirect user here for IdP logout
            }
        )

    except Exception as e:
        logger.exception("SSO logout failed")
        return error_response(safe_error_message(e, "logout"), status=500)


# =============================================================================
# Provider Configuration
# =============================================================================


# NOTE: List providers is a public endpoint - allows UI to show SSO buttons
# to unauthenticated users. RBAC protection via middleware.
@auth_rate_limit(
    requests_per_minute=30,
    limiter_name="auth_sso_list_providers",
    endpoint_name="SSO list providers",
)
async def handle_list_providers(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    List available SSO providers.

    GET /api/v1/auth/sso/providers
    """
    try:
        providers = []

        # Check which providers are configured
        provider_configs = [
            {
                "type": "oidc",
                "name": "OIDC/OAuth 2.0",
                "env_vars": ["OIDC_CLIENT_ID", "OIDC_ISSUER_URL"],
            },
            {
                "type": "google",
                "name": "Google",
                "env_vars": ["GOOGLE_CLIENT_ID"],
            },
            {
                "type": "github",
                "name": "GitHub",
                "env_vars": ["GITHUB_CLIENT_ID"],
            },
            {
                "type": "azure_ad",
                "name": "Azure AD",
                "env_vars": ["AZURE_AD_CLIENT_ID", "AZURE_AD_TENANT_ID"],
            },
        ]

        for config in provider_configs:
            is_configured = all(os.environ.get(var) for var in config["env_vars"])
            providers.append(
                {
                    "type": config["type"],
                    "name": config["name"],
                    "enabled": is_configured,
                }
            )

        return success_response(
            {
                "providers": providers,
                "sso_enabled": any(p["enabled"] for p in providers),
            }
        )

    except Exception as e:
        logger.exception("Failed to list providers")
        return error_response(safe_error_message(e, "list SSO providers"), status=500)


@require_permission("auth:manage_sso")
@auth_rate_limit(
    requests_per_minute=20,
    limiter_name="auth_sso_config",
    endpoint_name="SSO config",
)
async def handle_get_sso_config(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get SSO configuration for a provider.

    GET /api/v1/auth/sso/config
    Query params:
        provider: str
    """
    try:
        provider_type = data.get("provider", "oidc")

        # Return public configuration (no secrets)
        config: dict[str, Any] = {
            "provider": provider_type,
            "enabled": False,
        }

        if provider_type == "oidc":
            client_id = os.environ.get("OIDC_CLIENT_ID")
            issuer_url = os.environ.get("OIDC_ISSUER_URL")

            if client_id and issuer_url:
                config.update(
                    {
                        "enabled": True,
                        "issuer_url": issuer_url,
                        "scopes": os.environ.get("OIDC_SCOPES", "openid,email,profile").split(","),
                    }
                )

        elif provider_type == "google":
            client_id = os.environ.get("GOOGLE_CLIENT_ID")
            if client_id:
                config.update(
                    {
                        "enabled": True,
                        "issuer_url": "https://accounts.google.com",
                        "scopes": ["openid", "email", "profile"],
                    }
                )

        elif provider_type == "github":
            client_id = os.environ.get("GITHUB_CLIENT_ID")
            if client_id:
                config.update(
                    {
                        "enabled": True,
                        "authorization_endpoint": "https://github.com/login/oauth/authorize",
                        "scopes": ["user:email", "read:user"],
                    }
                )

        return success_response(config)

    except Exception as e:
        logger.exception("Failed to get SSO config")
        return error_response(safe_error_message(e, "get SSO config"), status=500)


# =============================================================================
# Handler Registration
# =============================================================================


def get_sso_handlers() -> dict[str, Any]:
    """Get all SSO handlers for registration."""
    return {
        "sso_login": handle_sso_login,
        "sso_callback": handle_sso_callback,
        "sso_refresh": handle_sso_refresh,
        "sso_logout": handle_sso_logout,
        "sso_list_providers": handle_list_providers,
        "sso_get_config": handle_get_sso_config,
    }


__all__ = [
    "handle_sso_login",
    "handle_sso_callback",
    "handle_sso_refresh",
    "handle_sso_logout",
    "handle_list_providers",
    "handle_get_sso_config",
    "get_sso_handlers",
    # Internal but exported for testing
    "_get_sso_state_store",
    "_sso_providers",
    "_sso_providers_lock",
    "AUTH_SESSION_TTL",
    "_get_sso_provider",
]
