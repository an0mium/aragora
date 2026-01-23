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
import secrets
import threading
import time
from typing import Any, Dict

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# Thread-safe SSO provider instances
_sso_providers: Dict[str, Any] = {}
_sso_providers_lock = threading.Lock()

# Session store for state -> session data
_auth_sessions: Dict[str, Dict[str, Any]] = {}
_auth_sessions_lock = threading.Lock()

# Session TTL (10 minutes)
AUTH_SESSION_TTL = 600


def _get_sso_provider(provider_type: str = "oidc"):
    """Get or create SSO provider for type."""
    with _sso_providers_lock:
        if provider_type in _sso_providers:
            return _sso_providers[provider_type]

        try:
            if provider_type == "oidc":
                from aragora.auth.oidc import OIDCConfig, OIDCProvider

                # Get configuration from environment
                config = OIDCConfig(
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

                config = OIDCConfig(
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


def _cleanup_expired_sessions():
    """Remove expired auth sessions."""
    now = time.time()
    expired = []

    with _auth_sessions_lock:
        for state, session in _auth_sessions.items():
            if now - session.get("created_at", 0) > AUTH_SESSION_TTL:
                expired.append(state)

        for state in expired:
            del _auth_sessions[state]


# =============================================================================
# SSO Login Flow
# =============================================================================


async def handle_sso_login(
    data: Dict[str, Any],
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

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)

        # Store session data
        with _auth_sessions_lock:
            _auth_sessions[state] = {
                "provider_type": provider_type,
                "redirect_url": redirect_url,
                "created_at": time.time(),
            }

        # Cleanup old sessions periodically
        if len(_auth_sessions) % 10 == 0:
            _cleanup_expired_sessions()

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
        return error_response(f"SSO login failed: {str(e)}", status=500)


async def handle_sso_callback(
    data: Dict[str, Any],
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

        # Validate state and get session
        with _auth_sessions_lock:
            session = _auth_sessions.pop(state, None)

        if not session:
            return error_response("Invalid or expired state", status=401)

        # Check session expiry
        if time.time() - session.get("created_at", 0) > AUTH_SESSION_TTL:
            return error_response("Session expired", status=401)

        provider_type = session.get("provider_type", "oidc")
        redirect_url = session.get("redirect_url", "/")

        provider = _get_sso_provider(provider_type)
        if not provider:
            return error_response("SSO provider not available", status=503)

        # Authenticate with the provider
        sso_user = await provider.authenticate(code=code, state=state)

        # Create or update user in our system
        from aragora.billing.jwt_auth import create_access_token

        # Generate JWT token for our system
        token_data = {
            "sub": sso_user.id,
            "email": sso_user.email,
            "name": sso_user.full_name,
            "provider": provider_type,
            "roles": sso_user.roles,
        }
        access_token = create_access_token(token_data)

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
        return error_response(f"SSO authentication failed: {str(e)}", status=401)


async def handle_sso_refresh(
    data: Dict[str, Any],
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
        return error_response(f"Token refresh failed: {str(e)}", status=401)


async def handle_sso_logout(
    data: Dict[str, Any],
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
        return error_response(f"Logout failed: {str(e)}", status=500)


# =============================================================================
# Provider Configuration
# =============================================================================


async def handle_list_providers(
    data: Dict[str, Any],
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
        return error_response(f"List providers failed: {str(e)}", status=500)


async def handle_get_sso_config(
    data: Dict[str, Any],
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
        config: Dict[str, Any] = {
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
        return error_response(f"Get config failed: {str(e)}", status=500)


# =============================================================================
# Handler Registration
# =============================================================================


def get_sso_handlers() -> Dict[str, Any]:
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
]
