"""
Account management mixin.

Provides OAuth account linking, unlinking, and provider listing methods.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

from aragora.audit.unified import audit_action
from aragora.server.handlers.base import HandlerResult, error_response, handle_errors, json_response
from aragora.server.handlers.oauth.models import _get_param

from .utils import _impl

logger = logging.getLogger(__name__)


class AccountManagementMixin:
    """Mixin providing OAuth account management methods.

    Note: This mixin expects to be combined with a class that implements
    OAuthHandlerProtocol (i.e., OAuthHandler).
    """

    # Declare methods from parent class to satisfy mypy
    _get_user_store: Any
    _check_permission: Any
    read_json_body: Any

    # Provider-specific auth start methods
    _handle_google_auth_start: Any
    _handle_github_auth_start: Any
    _handle_microsoft_auth_start: Any
    _handle_apple_auth_start: Any
    _handle_oidc_auth_start: Any

    # Provider-specific callback methods
    _handle_google_callback: Any
    _handle_github_callback: Any
    _handle_microsoft_callback: Any
    _handle_apple_callback: Any
    _handle_oidc_callback: Any

    @handle_errors("list OAuth providers")
    def _handle_list_providers(self, handler: Any) -> HandlerResult:
        """List configured OAuth providers."""
        impl = _impl()
        providers = []

        if impl._get_google_client_id():
            providers.append(
                {
                    "id": "google",
                    "name": "Google",
                    "enabled": True,
                    "auth_url": "/api/auth/oauth/google",
                }
            )

        if impl._get_github_client_id():
            providers.append(
                {
                    "id": "github",
                    "name": "GitHub",
                    "enabled": True,
                    "auth_url": "/api/auth/oauth/github",
                }
            )

        if impl._get_microsoft_client_id():
            providers.append(
                {
                    "id": "microsoft",
                    "name": "Microsoft",
                    "enabled": True,
                    "auth_url": "/api/auth/oauth/microsoft",
                }
            )

        if impl._get_apple_client_id():
            providers.append(
                {
                    "id": "apple",
                    "name": "Apple",
                    "enabled": True,
                    "auth_url": "/api/auth/oauth/apple",
                }
            )

        if impl._get_oidc_issuer() and impl._get_oidc_client_id():
            providers.append(
                {
                    "id": "oidc",
                    "name": "SSO",
                    "enabled": True,
                    "auth_url": "/api/auth/oauth/oidc",
                }
            )

        return json_response({"providers": providers})

    @handle_errors("get OAuth authorization URL")
    def _handle_oauth_url(self, handler: Any, query_params: dict) -> HandlerResult:
        """Return OAuth authorization URL for a provider without redirecting."""
        provider = _get_param(query_params, "provider")
        if not provider:
            return error_response("Provider is required", 400)
        provider = provider.lower()

        provider_map = {
            "google": self._handle_google_auth_start,
            "github": self._handle_github_auth_start,
            "microsoft": self._handle_microsoft_auth_start,
            "apple": self._handle_apple_auth_start,
            "oidc": self._handle_oidc_auth_start,
        }
        handler_fn = provider_map.get(provider)
        if handler_fn is None:
            return error_response("Unsupported provider", 400)

        result = handler_fn(handler, query_params)
        auth_url = result.headers.get("Location") if result and result.headers else None
        if not auth_url:
            return error_response("Failed to generate OAuth URL", 500)

        from urllib.parse import parse_qs, urlparse

        state = None
        try:
            parsed = urlparse(auth_url)
            state_vals = parse_qs(parsed.query).get("state")
            if state_vals:
                state = state_vals[0]
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.warning("Failed to parse state from OAuth URL: %s", e)
            state = None

        return json_response({"auth_url": auth_url, "state": state})

    @handle_errors("OAuth callback (API)")
    def _handle_oauth_callback_api(self, handler: Any) -> HandlerResult:
        """Complete OAuth flow and return tokens as JSON."""
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        provider = body.get("provider", "").lower()
        code = body.get("code")
        state = body.get("state")

        if not provider or not code or not state:
            return error_response("provider, code, and state are required", 400)

        callback_map = {
            "google": self._handle_google_callback,
            "github": self._handle_github_callback,
            "microsoft": self._handle_microsoft_callback,
            "apple": self._handle_apple_callback,
            "oidc": self._handle_oidc_callback,
        }
        handler_fn = callback_map.get(provider)
        if handler_fn is None:
            return error_response("Unsupported provider", 400)

        result = handler_fn(handler, {"code": code, "state": state})
        location = result.headers.get("Location") if result and result.headers else None
        if not location:
            return error_response("OAuth callback did not return redirect", 502)

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(location)
        # Tokens are in URL fragment (#) for security (not logged by servers/proxies)
        # Fall back to query params (?) for backward compatibility
        params = parse_qs(parsed.fragment) if parsed.fragment else parse_qs(parsed.query)
        if "error" in params:
            return error_response(params.get("error", ["OAuth error"])[0], 400)

        access_token = params.get("access_token", [None])[0]
        if not access_token:
            # Log to help debug token location issues
            logger.warning(
                "OAuth callback missing tokens: fragment=%s, query=%s, location_prefix=%s...",
                bool(parsed.fragment),
                bool(parsed.query),
                location[:50],
            )
            return error_response("OAuth callback did not return tokens", 502)

        refresh_token = params.get("refresh_token", [None])[0]
        token_type = params.get("token_type", ["Bearer"])[0]
        expires_in_val = params.get("expires_in", [None])[0]
        try:
            expires_in = int(expires_in_val) if expires_in_val is not None else None
        except (TypeError, ValueError):
            expires_in = None

        return json_response(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": token_type,
                "expires_in": expires_in,
            }
        )

    @handle_errors("get user OAuth providers")
    def _handle_get_user_providers(self, handler: Any) -> HandlerResult:
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
    def _handle_link_account(self, handler: Any) -> HandlerResult:
        """Link OAuth account to current user (initiated from settings)."""
        impl = _impl()

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
        redirect_url = body.get("redirect_url", impl._get_oauth_success_url())

        # Validate redirect URL against allowlist (same as start flow)
        if not impl._validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL. Only approved domains are allowed.", 400)

        state = impl._generate_state(user_id=auth_ctx.user_id, redirect_url=redirect_url)

        if provider == "google":
            if not impl.GOOGLE_CLIENT_ID:
                return error_response("Google OAuth not configured", 503)
            params = {
                "client_id": impl.GOOGLE_CLIENT_ID,
                "redirect_uri": impl._get_google_redirect_uri(),
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
                "access_type": "offline",
                "prompt": "consent",
            }
            auth_url = f"{impl.GOOGLE_AUTH_URL}?{urlencode(params)}"
            return json_response({"auth_url": auth_url})

        if provider == "github":
            if not impl.GITHUB_CLIENT_ID:
                return error_response("GitHub OAuth not configured", 503)
            params = {
                "client_id": impl.GITHUB_CLIENT_ID,
                "redirect_uri": impl._get_github_redirect_uri(),
                "scope": "read:user user:email",
                "state": state,
            }
            auth_url = f"{impl.GITHUB_AUTH_URL}?{urlencode(params)}"
            return json_response({"auth_url": auth_url})

        if provider == "microsoft":
            microsoft_client_id = impl._get_microsoft_client_id()
            if not microsoft_client_id:
                return error_response("Microsoft OAuth not configured", 503)
            tenant = impl._get_microsoft_tenant()
            auth_url_base = impl.MICROSOFT_AUTH_URL_TEMPLATE.format(tenant=tenant)
            params = {
                "client_id": microsoft_client_id,
                "redirect_uri": impl._get_microsoft_redirect_uri(),
                "response_type": "code",
                "scope": "openid email profile User.Read",
                "state": state,
                "response_mode": "query",
            }
            auth_url = f"{auth_url_base}?{urlencode(params)}"
            return json_response({"auth_url": auth_url})

        if provider == "apple":
            apple_client_id = impl._get_apple_client_id()
            if not apple_client_id:
                return error_response("Apple OAuth not configured", 503)
            params = {
                "client_id": apple_client_id,
                "redirect_uri": impl._get_apple_redirect_uri(),
                "response_type": "code id_token",
                "scope": "name email",
                "state": state,
                "response_mode": "form_post",
            }
            auth_url = f"{impl.APPLE_AUTH_URL}?{urlencode(params)}"
            return json_response({"auth_url": auth_url})

        if provider == "oidc":
            oidc_issuer = impl._get_oidc_issuer()
            oidc_client_id = impl._get_oidc_client_id()
            if not oidc_issuer or not oidc_client_id:
                return error_response("OIDC provider not configured", 503)
            # Fetch discovery document synchronously for auth endpoint
            discovery = self._get_oidc_discovery(oidc_issuer)
            if not discovery or not discovery.get("authorization_endpoint"):
                return error_response("OIDC discovery failed", 503)
            params = {
                "client_id": oidc_client_id,
                "redirect_uri": impl._get_oidc_redirect_uri(),
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
            }
            auth_url = f"{discovery['authorization_endpoint']}?{urlencode(params)}"
            return json_response({"auth_url": auth_url})

        return error_response("Unsupported provider", 400)

    @handle_errors("unlink OAuth account")
    def _handle_unlink_account(self, handler: Any) -> HandlerResult:
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

        logger.info("Unlinked %s from user %s", provider, auth_ctx.user_id)

        # Audit OAuth unlink
        audit_action(
            user_id=auth_ctx.user_id,
            action="oauth_unlink",
            resource_type="auth",
            resource_id=auth_ctx.user_id,
            provider=provider,
        )

        return json_response({"message": f"Unlinked {provider} successfully"})
