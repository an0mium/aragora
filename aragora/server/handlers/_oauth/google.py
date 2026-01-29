"""
Google OAuth mixin.

Provides Google OAuth 2.0 authentication methods for the OAuthHandler.
"""

from __future__ import annotations

import json
import logging
import time
from urllib.parse import urlencode

from aragora.audit.unified import audit_action, audit_security
from aragora.server.handlers.base import HandlerResult, error_response, handle_errors, log_request
from aragora.server.handlers.oauth.models import OAuthUserInfo, _get_param

from .utils import _impl

logger = logging.getLogger(__name__)


class GoogleOAuthMixin:
    """Mixin providing Google OAuth 2.0 methods."""

    @handle_errors("Google OAuth start")
    @log_request("Google OAuth start")
    def _handle_google_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to Google OAuth consent screen."""
        impl = _impl()
        google_client_id = impl._get_google_client_id()
        if not google_client_id:
            return error_response("Google OAuth not configured", 503)

        # Get optional redirect URL from query params
        oauth_success_url = impl._get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        # Security: Validate redirect URL against allowlist to prevent open redirects
        if not impl._validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL. Only approved domains are allowed.", 400)

        # Check if this is for account linking (user already authenticated)
        user_id = None
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if auth_ctx.is_authenticated:
            user_id = auth_ctx.user_id

        # Generate state for CSRF protection
        state = impl._generate_state(user_id=user_id, redirect_url=redirect_url)

        # Build authorization URL
        params = {
            "client_id": google_client_id,
            "redirect_uri": impl._get_google_redirect_uri(),
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{impl.GOOGLE_AUTH_URL}?{urlencode(params)}"

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
        impl = _impl()

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
        state_data = impl._validate_state(state)
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
        redirect_url = state_data.get("redirect_url", impl._get_oauth_success_url())
        logger.info(f"OAuth callback: redirecting to {redirect_url}")
        return self._redirect_with_tokens(redirect_url, tokens)

    def _exchange_code_for_tokens(self, code: str) -> dict:
        """Exchange authorization code for access tokens."""
        import urllib.error
        import urllib.request

        impl = _impl()
        data = urlencode(
            {
                "code": code,
                "client_id": impl.GOOGLE_CLIENT_ID,
                "client_secret": impl._get_google_client_secret(),
                "redirect_uri": impl._get_google_redirect_uri(),
                "grant_type": "authorization_code",
            }
        ).encode()

        req = urllib.request.Request(
            impl.GOOGLE_TOKEN_URL,
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

        impl = _impl()
        req = urllib.request.Request(
            impl.GOOGLE_USERINFO_URL,
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
