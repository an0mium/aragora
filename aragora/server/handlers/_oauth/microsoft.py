"""
Microsoft OAuth mixin.

Provides Microsoft OAuth (Azure AD) authentication methods for the OAuthHandler.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any
from collections.abc import Coroutine
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import httpx

from aragora.server.handlers.base import HandlerResult, error_response, handle_errors, log_request
from aragora.server.handlers.oauth.models import OAuthUserInfo, _get_param

from .utils import _impl, _maybe_await

logger = logging.getLogger(__name__)


class MicrosoftOAuthMixin:
    """Mixin providing Microsoft OAuth methods.

    Note: This mixin expects to be combined with a class that implements
    OAuthHandlerProtocol (i.e., OAuthHandler).
    """

    # Declare methods from parent class to satisfy mypy
    _get_user_store: Any
    _redirect_with_error: Any
    _complete_oauth_flow: Any

    @handle_errors("Microsoft OAuth start")
    @log_request("Microsoft OAuth start")
    def _handle_microsoft_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to Microsoft OAuth consent screen."""
        impl = _impl()
        microsoft_client_id = impl._get_microsoft_client_id()
        if not microsoft_client_id:
            return error_response("Microsoft OAuth not configured", 503)

        oauth_success_url = impl._get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        if not impl._validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL", 400)

        user_id = None
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if auth_ctx.is_authenticated:
            user_id = auth_ctx.user_id

        state = impl._generate_state(user_id=user_id, redirect_url=redirect_url)

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

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={auth_url}"></head></html>'.encode(),
            headers={"Location": auth_url},
        )

    @handle_errors("Microsoft OAuth callback")
    @log_request("Microsoft OAuth callback")
    async def _handle_microsoft_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle Microsoft OAuth callback."""
        impl = _impl()

        error = _get_param(query_params, "error")
        if error:
            error_desc = _get_param(query_params, "error_description", error)
            logger.warning("Microsoft OAuth error: %s - %s", error, error_desc)
            return self._redirect_with_error(f"OAuth error: {error_desc}")

        state = _get_param(query_params, "state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = impl._validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        code = _get_param(query_params, "code")
        if not code:
            return self._redirect_with_error("Missing authorization code")

        try:
            token_data = self._exchange_microsoft_code(code)
            if inspect.isawaitable(token_data):
                token_data = await token_data
        except (
            httpx.HTTPError,
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            logger.error("Microsoft token exchange failed: %s", e)
            return self._redirect_with_error("Failed to exchange authorization code")

        access_token = token_data.get("access_token")
        if not access_token:
            return self._redirect_with_error("No access token received")

        try:
            user_info = self._get_microsoft_user_info(access_token)
            if inspect.isawaitable(user_info):
                user_info = await user_info
        except (
            httpx.HTTPError,
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as e:
            logger.error("Failed to get Microsoft user info: %s", e)
            return self._redirect_with_error("Failed to get user info")

        return await _maybe_await(self._complete_oauth_flow(user_info, state_data))

    def _exchange_microsoft_code(self, code: str) -> dict | Coroutine[Any, Any, dict]:
        """Exchange Microsoft authorization code for access token."""
        impl = _impl()
        tenant = impl._get_microsoft_tenant()
        token_url = impl.MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant=tenant)

        data = {
            "code": code,
            "client_id": impl._get_microsoft_client_id(),
            "client_secret": impl._get_microsoft_client_secret(),
            "redirect_uri": impl._get_microsoft_redirect_uri(),
            "grant_type": "authorization_code",
        }
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            encoded = urlencode(data).encode("utf-8")
            req = Request(  # noqa: S310 -- hardcoded Microsoft OAuth URL
                token_url,
                data=encoded,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urlopen(req) as response:  # noqa: S310 -- hardcoded Microsoft OAuth URL
                body = response.read()
            return json.loads(body.decode("utf-8")) if body else {}

        async def _exchange_async() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                return response.json()

        return _exchange_async()

    def _get_microsoft_user_info(
        self, access_token: str
    ) -> OAuthUserInfo | Coroutine[Any, Any, OAuthUserInfo]:
        """Get user info from Microsoft Graph API."""
        impl = _impl()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            req = Request(  # noqa: S310 -- hardcoded Microsoft API URL
                impl.MICROSOFT_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            with urlopen(req) as response:  # noqa: S310 -- hardcoded Microsoft API URL
                body = response.read()
            user_data = json.loads(body.decode("utf-8")) if body else {}

            email = user_data.get("mail") or user_data.get("userPrincipalName", "")
            if not email or "@" not in email:
                raise ValueError("Could not retrieve email from Microsoft")

            provider_user_id = user_data.get("id")
            if not provider_user_id:
                raise ValueError("Microsoft user response missing 'id' field")

            return OAuthUserInfo(
                provider="microsoft",
                provider_user_id=provider_user_id,
                email=email,
                name=user_data.get("displayName", email.split("@")[0]),
                picture=None,  # Microsoft Graph requires separate call for photo
                email_verified=True,  # Microsoft validates emails
            )

        async def _userinfo_async() -> OAuthUserInfo:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    impl.MICROSOFT_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                user_data = response.json()

            email = user_data.get("mail") or user_data.get("userPrincipalName", "")
            if not email or "@" not in email:
                raise ValueError("Could not retrieve email from Microsoft")

            provider_user_id = user_data.get("id")
            if not provider_user_id:
                raise ValueError("Microsoft user response missing 'id' field")

            return OAuthUserInfo(
                provider="microsoft",
                provider_user_id=provider_user_id,
                email=email,
                name=user_data.get("displayName", email.split("@")[0]),
                picture=None,  # Microsoft Graph requires separate call for photo
                email_verified=True,  # Microsoft validates emails
            )

        return _userinfo_async()
