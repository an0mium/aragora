"""
OAuth configuration functions.

Provides lazy-loaded configuration from AWS Secrets Manager or environment
variables. All config functions are called at runtime to support dynamic
secret rotation.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# =============================================================================
# Core configuration helpers
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


# =============================================================================
# Provider-specific client credentials
# =============================================================================


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


# =============================================================================
# Redirect URIs (with dev fallbacks)
# =============================================================================


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


# =============================================================================
# Frontend URLs
# =============================================================================


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


# =============================================================================
# Legacy module-level variables (for backward compatibility)
# =============================================================================

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


# =============================================================================
# Configuration validation
# =============================================================================


def validate_oauth_config(log_warnings: bool = True) -> list[str]:
    """
    Validate OAuth configuration and return list of missing required vars.

    Call this at startup to catch configuration errors early.
    Returns empty list if configuration is valid, or list of missing var names.

    Args:
        log_warnings: If True, log warnings for missing configuration
    """
    import sys

    missing = []
    running_under_pytest = "pytest" in sys.modules

    # Check JWT secret (required for all OAuth flows)
    # Skip in pytest since the JWT module generates ephemeral secrets for tests
    if not running_under_pytest:
        jwt_secret = os.environ.get("ARAGORA_JWT_SECRET", "")
        if not jwt_secret:
            missing.append("ARAGORA_JWT_SECRET")
            if log_warnings:
                logger.warning(
                    "OAuth config: ARAGORA_JWT_SECRET is not set! "
                    "OAuth login will fail. Generate with: "
                    'python -c "import secrets; print(secrets.token_urlsafe(32))"'
                )
        elif len(jwt_secret) < 32:
            missing.append("ARAGORA_JWT_SECRET (too short, need 32+ chars)")
            if log_warnings:
                logger.warning(
                    "OAuth config: ARAGORA_JWT_SECRET is too short "
                    f"({len(jwt_secret)} chars). Need at least 32 characters."
                )

    if not _IS_PRODUCTION:
        return missing  # Only check JWT in dev mode, skip provider checks

    # If Google OAuth is enabled (client ID set), check required vars
    if GOOGLE_CLIENT_ID:
        if not _get_google_client_secret():
            missing.append("GOOGLE_OAUTH_CLIENT_SECRET")
        if not _get_google_redirect_uri():
            missing.append("GOOGLE_OAUTH_REDIRECT_URI")
        if not _get_oauth_success_url():
            missing.append("OAUTH_SUCCESS_URL")
        if not _get_oauth_error_url():
            missing.append("OAUTH_ERROR_URL")
        if not ALLOWED_OAUTH_REDIRECT_HOSTS:
            missing.append("OAUTH_ALLOWED_REDIRECT_HOSTS")

    # If GitHub OAuth is enabled (client ID set), check required vars
    if GITHUB_CLIENT_ID:
        if not _get_github_client_secret():
            missing.append("GITHUB_OAUTH_CLIENT_SECRET")
        if not _get_github_redirect_uri():
            missing.append("GITHUB_OAUTH_REDIRECT_URI")
        # Shared URLs only need to be set once
        if not _get_oauth_success_url() and "OAUTH_SUCCESS_URL" not in missing:
            missing.append("OAUTH_SUCCESS_URL")
        if not _get_oauth_error_url() and "OAUTH_ERROR_URL" not in missing:
            missing.append("OAUTH_ERROR_URL")
        if not ALLOWED_OAUTH_REDIRECT_HOSTS and "OAUTH_ALLOWED_REDIRECT_HOSTS" not in missing:
            missing.append("OAUTH_ALLOWED_REDIRECT_HOSTS")

    if missing and log_warnings:
        logger.warning(f"OAuth config: Missing required variables: {missing}")

    return missing


def get_oauth_config_status() -> dict[str, Any]:
    """
    Get detailed OAuth configuration status for diagnostics.

    Returns a dict with configuration status for each provider and required
    settings. Sensitive values are masked.
    """

    def _mask(val: str) -> str:
        if not val:
            return "(not set)"
        if len(val) <= 8:
            return "****"
        return f"{val[:4]}...{val[-4:]}"

    jwt_secret = os.environ.get("ARAGORA_JWT_SECRET", "")

    return {
        "environment": {
            "is_production": _IS_PRODUCTION,
            "aragora_env": os.environ.get("ARAGORA_ENV", "(not set)"),
        },
        "jwt": {
            "secret_configured": bool(jwt_secret),
            "secret_length": len(jwt_secret) if jwt_secret else 0,
            "secret_valid": len(jwt_secret) >= 32,
        },
        "google": {
            "client_id": _mask(GOOGLE_CLIENT_ID),
            "client_secret_set": bool(_get_google_client_secret()),
            "redirect_uri": _get_google_redirect_uri() or "(not set)",
            "enabled": bool(GOOGLE_CLIENT_ID),
        },
        "github": {
            "client_id": _mask(GITHUB_CLIENT_ID),
            "client_secret_set": bool(_get_github_client_secret()),
            "redirect_uri": _get_github_redirect_uri() or "(not set)",
            "enabled": bool(GITHUB_CLIENT_ID),
        },
        "urls": {
            "success_url": _get_oauth_success_url() or "(not set)",
            "error_url": _get_oauth_error_url() or "(not set)",
            "allowed_redirect_hosts": list(ALLOWED_OAUTH_REDIRECT_HOSTS) or ["(none)"],
        },
        "validation": {
            "missing_vars": validate_oauth_config(log_warnings=False),
            "is_valid": len(validate_oauth_config(log_warnings=False)) == 0,
        },
    }


# =============================================================================
# Provider endpoint constants
# =============================================================================

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
MICROSOFT_AUTH_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
MICROSOFT_USERINFO_URL = "https://graph.microsoft.com/v1.0/me"

# Apple OAuth endpoints
APPLE_AUTH_URL = "https://appleid.apple.com/auth/authorize"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"
