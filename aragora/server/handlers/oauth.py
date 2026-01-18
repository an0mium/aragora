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

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from .utils.rate_limit import RateLimiter, get_client_ip

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
    return frozenset(
        host.strip().lower() for host in val.split(",") if host.strip()
    )


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


class OAuthHandler(BaseHandler):
    """Handler for OAuth authentication endpoints."""

    ROUTES = [
        "/api/auth/oauth/google",
        "/api/auth/oauth/google/callback",
        "/api/auth/oauth/github",
        "/api/auth/oauth/github/callback",
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

        if path == "/api/auth/oauth/google" and method == "GET":
            return self._handle_google_auth_start(handler, query_params)

        if path == "/api/auth/oauth/google/callback" and method == "GET":
            return self._handle_google_callback(handler, query_params)

        if path == "/api/auth/oauth/github" and method == "GET":
            return self._handle_github_auth_start(handler, query_params)

        if path == "/api/auth/oauth/github/callback" and method == "GET":
            return self._handle_github_callback(handler, query_params)

        if path == "/api/auth/oauth/link" and method == "POST":
            return self._handle_link_account(handler)

        if path == "/api/auth/oauth/unlink" and method == "DELETE":
            return self._handle_unlink_account(handler)

        if path == "/api/auth/oauth/providers" and method == "GET":
            return self._handle_list_providers(handler)

        if path == "/api/user/oauth-providers" and method == "GET":
            return self._handle_get_user_providers(handler)

        return error_response("Method not allowed", 405)

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    @handle_errors("Google OAuth start")
    @log_request("Google OAuth start")
    def _handle_google_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to Google OAuth consent screen."""
        google_client_id = _get_google_client_id()
        if not google_client_id:
            return error_response("Google OAuth not configured", 503)

        # Get optional redirect URL from query params
        oauth_success_url = _get_oauth_success_url()
        redirect_url = query_params.get("redirect_url", [oauth_success_url])[0]

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
        error = query_params.get("error", [None])[0]
        if error:
            error_desc = query_params.get("error_description", [error])[0]
            logger.warning(f"Google OAuth error: {error} - {error_desc}")
            return self._redirect_with_error(f"OAuth error: {error_desc}")

        # Validate state
        state = query_params.get("state", [None])[0]
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = _validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        # Get authorization code
        code = query_params.get("code", [None])[0]
        if not code:
            return self._redirect_with_error("Missing authorization code")

        # Exchange code for tokens
        try:
            token_data = self._exchange_code_for_tokens(code)
        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            return self._redirect_with_error("Failed to exchange authorization code")

        access_token = token_data.get("access_token")
        if not access_token:
            return self._redirect_with_error("No access token received")

        # Get user info from Google
        try:
            user_info = self._get_google_user_info(access_token)
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return self._redirect_with_error("Failed to get user info from Google")

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

        logger.info(f"OAuth login: {user.email} via Google")

        # Redirect to frontend with tokens
        redirect_url = state_data.get("redirect_url", _get_oauth_success_url())
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
        redirect_url = query_params.get("redirect_url", [oauth_success_url])[0]

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
        error = query_params.get("error", [None])[0]
        if error:
            error_desc = query_params.get("error_description", [error])[0]
            logger.warning(f"GitHub OAuth error: {error} - {error_desc}")
            return self._redirect_with_error(f"OAuth error: {error_desc}")

        # Validate state
        state = query_params.get("state", [None])[0]
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = _validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        # Get authorization code
        code = query_params.get("code", [None])[0]
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

    def _redirect_with_tokens(self, redirect_url: str, tokens) -> HandlerResult:
        """Redirect to frontend with tokens in URL fragment."""
        # Use URL fragment for tokens (more secure than query params)
        fragment = urlencode(
            {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "token_type": "Bearer",
                "expires_in": tokens.expires_in,
            }
        )
        url = f"{redirect_url}#{fragment}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={url}"></head></html>'.encode(),
            headers={"Location": url},
        )

    def _redirect_with_error(self, error: str) -> HandlerResult:
        """Redirect to error page with error message."""
        from urllib.parse import quote

        url = f"{_get_oauth_error_url()}?error={quote(error)}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={url}"></head></html>'.encode(),
            headers={"Location": url},
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
                    "auth_url": "/api/auth/oauth/google",
                }
            )

        if _get_github_client_id():
            providers.append(
                {
                    "id": "github",
                    "name": "GitHub",
                    "enabled": True,
                    "auth_url": "/api/auth/oauth/github",
                }
            )

        return json_response({"providers": providers})

    @handle_errors("get user OAuth providers")
    def _handle_get_user_providers(self, handler) -> HandlerResult:
        """Get OAuth providers linked to the current user."""
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

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
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        provider = body.get("provider", "").lower()
        if provider not in ["google", "github"]:
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
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        provider = body.get("provider", "").lower()
        if provider not in ["google", "github"]:
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

        return json_response({"message": f"Unlinked {provider} successfully"})


__all__ = ["OAuthHandler", "validate_oauth_config"]
