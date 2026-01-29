"""
GitHub OAuth mixin.

Provides GitHub OAuth authentication methods for the OAuthHandler.
"""

from __future__ import annotations

import json
import logging
import time
from urllib.parse import urlencode

from aragora.server.handlers.base import HandlerResult, error_response, handle_errors, log_request
from aragora.server.handlers.oauth.models import OAuthUserInfo, _get_param

from .utils import _impl

logger = logging.getLogger(__name__)


class GitHubOAuthMixin:
    """Mixin providing GitHub OAuth methods."""

    @handle_errors("GitHub OAuth start")
    @log_request("GitHub OAuth start")
    def _handle_github_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to GitHub OAuth consent screen."""
        impl = _impl()
        github_client_id = impl._get_github_client_id()
        if not github_client_id:
            return error_response("GitHub OAuth not configured", 503)

        # Get optional redirect URL from query params
        oauth_success_url = impl._get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        # Security: Validate redirect URL against allowlist
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
            "client_id": github_client_id,
            "redirect_uri": impl._get_github_redirect_uri(),
            "scope": "read:user user:email",
            "state": state,
        }
        auth_url = f"{impl.GITHUB_AUTH_URL}?{urlencode(params)}"

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
        impl = _impl()

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

        state_data = impl._validate_state(state)
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
        redirect_url = state_data.get("redirect_url", impl._get_oauth_success_url())
        return self._redirect_with_tokens(redirect_url, tokens)

    def _exchange_github_code(self, code: str) -> dict:
        """Exchange GitHub authorization code for access token."""
        import urllib.error
        import urllib.request

        impl = _impl()
        data = urlencode(
            {
                "code": code,
                "client_id": impl.GITHUB_CLIENT_ID,
                "client_secret": impl._get_github_client_secret(),
                "redirect_uri": impl._get_github_redirect_uri(),
            }
        ).encode()

        req = urllib.request.Request(
            impl.GITHUB_TOKEN_URL,
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

        impl = _impl()

        # Get basic user info
        req = urllib.request.Request(
            impl.GITHUB_USERINFO_URL,
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
                impl.GITHUB_EMAILS_URL,
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
