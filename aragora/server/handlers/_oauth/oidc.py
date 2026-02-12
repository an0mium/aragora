"""
Generic OIDC OAuth mixin.

Provides generic OpenID Connect authentication methods for the OAuthHandler.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any
from collections.abc import Coroutine
from urllib.parse import urlencode
from urllib.request import Request
import urllib.request as urllib_request

import httpx

from aragora.server.handlers.base import HandlerResult, error_response, handle_errors, log_request
from aragora.server.handlers.oauth.models import OAuthUserInfo, _get_param

from .utils import _impl, _maybe_await

logger = logging.getLogger(__name__)


class OIDCOAuthMixin:
    """Mixin providing generic OIDC authentication methods.

    Note: This mixin expects to be combined with a class that implements
    OAuthHandlerProtocol (i.e., OAuthHandler).
    """

    # Declare methods from parent class to satisfy mypy
    _get_user_store: Any
    _redirect_with_error: Any
    _complete_oauth_flow: Any

    @handle_errors("OIDC OAuth start")
    @log_request("OIDC OAuth start")
    async def _handle_oidc_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to generic OIDC provider."""
        impl = _impl()
        oidc_issuer = impl._get_oidc_issuer()
        oidc_client_id = impl._get_oidc_client_id()

        if not oidc_issuer or not oidc_client_id:
            return error_response("OIDC provider not configured", 503)

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

        # Discover OIDC endpoints
        discovery = self._get_oidc_discovery(oidc_issuer)
        if inspect.isawaitable(discovery):
            try:
                discovery = await asyncio.wait_for(discovery, timeout=10.0)
            except asyncio.TimeoutError:
                logger.error(f"OIDC discovery timed out for issuer: {oidc_issuer}")
                return error_response("OIDC provider discovery timed out", 504)
        auth_endpoint = discovery.get("authorization_endpoint")

        if not auth_endpoint:
            return error_response("OIDC discovery failed", 503)

        params = {
            "client_id": oidc_client_id,
            "redirect_uri": impl._get_oidc_redirect_uri(),
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
    async def _handle_oidc_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle generic OIDC callback."""
        impl = _impl()
        error = _get_param(query_params, "error")
        if error:
            error_desc = _get_param(query_params, "error_description", error)
            return self._redirect_with_error(f"OIDC error: {error_desc}")

        state = _get_param(query_params, "state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = impl._validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        code = _get_param(query_params, "code")
        if not code:
            return self._redirect_with_error("Missing authorization code")

        oidc_issuer = impl._get_oidc_issuer()
        discovery = self._get_oidc_discovery(oidc_issuer)

        try:
            token_data = self._exchange_oidc_code(code, discovery)  # type: ignore[arg-type]
            if inspect.isawaitable(token_data):
                token_data = await token_data
        except Exception as e:
            logger.error(f"OIDC token exchange failed: {e}")
            return self._redirect_with_error("Failed to exchange authorization code")

        access_token = token_data.get("access_token")
        id_token = token_data.get("id_token")

        try:
            user_info = self._get_oidc_user_info(access_token, id_token, discovery)  # type: ignore[arg-type]
            if inspect.isawaitable(user_info):
                user_info = await user_info
        except Exception as e:
            logger.error(f"Failed to get OIDC user info: {e}")
            return self._redirect_with_error("Failed to get user info")

        return await _maybe_await(self._complete_oauth_flow(user_info, state_data))

    def _get_oidc_discovery(self, issuer: str) -> dict | Coroutine[Any, Any, dict]:
        """Fetch OIDC discovery document."""
        discovery_url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                req = Request(discovery_url)
                with urllib_request.urlopen(req) as response:
                    body = response.read()
                return json.loads(body.decode("utf-8")) if body else {}
            except Exception as e:
                logger.error(f"OIDC discovery failed: {e}")
                return {}

        async def _discovery_async() -> dict:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(discovery_url)
                    return response.json()
            except Exception as e:
                logger.error(f"OIDC discovery failed: {e}")
                return {}

        return _discovery_async()

    def _exchange_oidc_code(self, code: str, discovery: dict) -> dict | Coroutine[Any, Any, dict]:
        """Exchange OIDC authorization code for tokens."""
        impl = _impl()
        token_endpoint = discovery.get("token_endpoint")
        if not token_endpoint:
            raise ValueError("No token endpoint in OIDC discovery")

        data = {
            "code": code,
            "client_id": impl._get_oidc_client_id(),
            "client_secret": impl._get_oidc_client_secret(),
            "redirect_uri": impl._get_oidc_redirect_uri(),
            "grant_type": "authorization_code",
        }

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            encoded = urlencode(data).encode("utf-8")
            req = Request(
                token_endpoint,
                data=encoded,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib_request.urlopen(req) as response:
                body = response.read()
            return json.loads(body.decode("utf-8")) if body else {}

        async def _exchange_async() -> dict:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    token_endpoint,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                return response.json()

        return _exchange_async()

    def _get_oidc_user_info(
        self, access_token: str, id_token: str, discovery: dict
    ) -> OAuthUserInfo | Coroutine[Any, Any, OAuthUserInfo]:
        """Get user info from OIDC userinfo endpoint or id_token."""
        import base64

        userinfo_endpoint = discovery.get("userinfo_endpoint")

        def _build_user(user_data: dict) -> OAuthUserInfo:
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

        def _fallback_id_token(user_data: dict) -> dict:
            if user_data or not id_token:
                return user_data
            parts = id_token.split(".")
            if len(parts) == 3:
                payload_b64 = parts[1]
                padding = 4 - len(payload_b64) % 4
                if padding != 4:
                    payload_b64 += "=" * padding
                return json.loads(base64.urlsafe_b64decode(payload_b64))
            return user_data

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            user_data: dict = {}
            if userinfo_endpoint and access_token:
                try:
                    req = Request(
                        userinfo_endpoint,
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    with urllib_request.urlopen(req) as response:
                        body = response.read()
                    user_data = json.loads(body.decode("utf-8")) if body else {}
                except Exception as e:
                    logger.warning(f"OIDC userinfo failed, falling back to id_token: {e}")

            user_data = _fallback_id_token(user_data)
            return _build_user(user_data)

        async def _userinfo_async() -> OAuthUserInfo:
            user_data: dict = {}
            if userinfo_endpoint and access_token:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(
                            userinfo_endpoint,
                            headers={"Authorization": f"Bearer {access_token}"},
                        )
                        user_data = response.json()
                except Exception as e:
                    logger.warning(f"OIDC userinfo failed, falling back to id_token: {e}")

            user_data = _fallback_id_token(user_data)
            return _build_user(user_data)

        return _userinfo_async()
