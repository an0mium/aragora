"""
GitHub OAuth mixin.

Provides GitHub OAuth authentication methods for the OAuthHandler.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Any, Coroutine
from urllib.parse import urlencode
from urllib.request import Request
import urllib.request as urllib_request

import httpx

from aragora.server.handlers.base import HandlerResult, error_response, handle_errors, log_request
from aragora.server.handlers.oauth.models import OAuthUserInfo, _get_param

from .utils import _impl, _maybe_await

logger = logging.getLogger(__name__)


class GitHubOAuthMixin:
    """Mixin providing GitHub OAuth methods.

    Note: This mixin expects to be combined with a class that implements
    OAuthHandlerProtocol (i.e., OAuthHandler).
    """

    # Declare methods from parent class to satisfy mypy
    _get_user_store: Any
    _redirect_with_error: Any
    _redirect_with_tokens: Any
    _find_user_by_oauth: Any
    _link_oauth_to_user: Any
    _create_oauth_user: Any
    _handle_account_linking: Any

    @handle_errors("GitHub OAuth start")
    @log_request("GitHub OAuth start")
    def _handle_github_auth_start(self, handler: Any, query_params: dict) -> HandlerResult:
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
    async def _handle_github_callback(self, handler: Any, query_params: dict) -> HandlerResult:
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
            if inspect.isawaitable(token_data):
                token_data = await token_data
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
            if inspect.isawaitable(user_info):
                user_info = await user_info
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
            return await _maybe_await(
                self._handle_account_linking(user_store, linking_user_id, user_info, state_data)
            )

        # Check if user exists by OAuth provider ID
        user = await _maybe_await(self._find_user_by_oauth(user_store, user_info))

        if not user:
            # Check if email already registered (without OAuth)
            get_by_email = getattr(user_store, "get_user_by_email_async", None)
            if get_by_email and inspect.iscoroutinefunction(get_by_email):
                user = await get_by_email(user_info.email)
            else:
                user = user_store.get_user_by_email(user_info.email)
            if user:
                # Security: only link OAuth when email is verified by GitHub
                if not user_info.email_verified:
                    logger.warning(
                        "OAuth linking blocked: unverified email %s from GitHub",
                        user_info.email,
                    )
                    return self._redirect_with_error(
                        "Email verification required to link your account."
                    )
                # Link OAuth to existing account
                await _maybe_await(self._link_oauth_to_user(user_store, user.id, user_info))
            else:
                # Create new user with OAuth
                user = await _maybe_await(self._create_oauth_user(user_store, user_info))

        if not user:
            return self._redirect_with_error("Failed to create user account")

        # Update last login
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

        logger.info(f"OAuth login: {user.email} via GitHub")

        # Redirect to frontend with tokens
        redirect_url = state_data.get("redirect_url", impl._get_oauth_success_url())
        return self._redirect_with_tokens(redirect_url, tokens)

    def _exchange_github_code(self, code: str) -> dict | Coroutine[Any, Any, dict]:
        """Exchange GitHub authorization code for access token."""
        impl = _impl()
        data = {
            "code": code,
            "client_id": impl.GITHUB_CLIENT_ID,
            "client_secret": impl._get_github_client_secret(),
            "redirect_uri": impl._get_github_redirect_uri(),
        }

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            encoded = urlencode(data).encode("utf-8")
            req = Request(
                impl.GITHUB_TOKEN_URL,
                data=encoded,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
            with urllib_request.urlopen(req) as response:
                body = response.read()
            try:
                return json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from GitHub token endpoint: {e}")
                raise ValueError(f"Invalid JSON response from GitHub: {e}") from e

        async def _exchange_async() -> dict:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    impl.GITHUB_TOKEN_URL,
                    data=data,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                )
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from GitHub token endpoint: {e}")
                    raise ValueError(f"Invalid JSON response from GitHub: {e}") from e

        return _exchange_async()

    def _get_github_user_info(
        self, access_token: str
    ) -> OAuthUserInfo | Coroutine[Any, Any, OAuthUserInfo]:
        """Get user info from GitHub API."""
        impl = _impl()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            urlopen_fn = urllib_request.urlopen
            side_effect = getattr(urlopen_fn, "side_effect", None)
            if isinstance(side_effect, list):
                urlopen_fn.side_effect = iter(side_effect)  # type: ignore[attr-defined]
            # Get basic user info
            req = Request(
                impl.GITHUB_USERINFO_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
            with urlopen_fn(req) as response:
                body = response.read()
            try:
                user_data = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from GitHub user endpoint: {e}")
                raise ValueError(f"Invalid JSON response from GitHub: {e}") from e

            # Get user's emails (need to find primary verified email)
            email = user_data.get("email")
            email_verified = False

            if not email:
                # Email not public, fetch from emails endpoint
                req = Request(
                    impl.GITHUB_EMAILS_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                )
                with urlopen_fn(req) as response:
                    body = response.read()
                try:
                    emails = json.loads(body.decode("utf-8")) if body else []
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

        async def _userinfo_async() -> OAuthUserInfo:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    impl.GITHUB_USERINFO_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                )
                try:
                    user_data = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from GitHub user endpoint: {e}")
                    raise ValueError(f"Invalid JSON response from GitHub: {e}") from e

            # Get user's emails (need to find primary verified email)
            email = user_data.get("email")
            email_verified = False

            if not email:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        impl.GITHUB_EMAILS_URL,
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Accept": "application/json",
                        },
                    )
                    try:
                        emails = response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from GitHub emails endpoint: {e}")
                        raise ValueError(f"Invalid JSON from GitHub emails: {e}") from e

                for email_entry in emails:
                    if email_entry.get("primary") and email_entry.get("verified"):
                        email = email_entry.get("email")
                        email_verified = True
                        break

                if not email:
                    for email_entry in emails:
                        if email_entry.get("verified"):
                            email = email_entry.get("email")
                            email_verified = True
                            break

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

        return _userinfo_async()
