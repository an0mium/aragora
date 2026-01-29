"""
Microsoft OAuth mixin.

Provides Microsoft OAuth (Azure AD) authentication methods for the OAuthHandler.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlencode

from aragora.server.handlers.base import HandlerResult, error_response, handle_errors, log_request
from aragora.server.handlers.oauth.config import (
    _get_microsoft_client_id,
    _get_microsoft_client_secret,
    _get_microsoft_redirect_uri,
    _get_microsoft_tenant,
    _get_oauth_success_url,
    MICROSOFT_AUTH_URL_TEMPLATE,
    MICROSOFT_TOKEN_URL_TEMPLATE,
    MICROSOFT_USERINFO_URL,
)
from aragora.server.handlers.oauth.models import OAuthUserInfo, _get_param

from .utils import _validate_redirect_url, _validate_state, _generate_state

logger = logging.getLogger(__name__)


class MicrosoftOAuthMixin:
    """Mixin providing Microsoft OAuth methods."""

    @handle_errors("Microsoft OAuth start")
    @log_request("Microsoft OAuth start")
    def _handle_microsoft_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to Microsoft OAuth consent screen."""
        microsoft_client_id = _get_microsoft_client_id()
        if not microsoft_client_id:
            return error_response("Microsoft OAuth not configured", 503)

        oauth_success_url = _get_oauth_success_url()
        redirect_url = _get_param(query_params, "redirect_url", oauth_success_url)

        if not _validate_redirect_url(redirect_url):
            return error_response("Invalid redirect URL", 400)

        user_id = None
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if auth_ctx.is_authenticated:
            user_id = auth_ctx.user_id

        state = _generate_state(user_id=user_id, redirect_url=redirect_url)

        tenant = _get_microsoft_tenant()
        auth_url_base = MICROSOFT_AUTH_URL_TEMPLATE.format(tenant=tenant)

        params = {
            "client_id": microsoft_client_id,
            "redirect_uri": _get_microsoft_redirect_uri(),
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
    def _handle_microsoft_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle Microsoft OAuth callback."""
        error = _get_param(query_params, "error")
        if error:
            error_desc = _get_param(query_params, "error_description", error)
            logger.warning(f"Microsoft OAuth error: {error} - {error_desc}")
            return self._redirect_with_error(f"OAuth error: {error_desc}")

        state = _get_param(query_params, "state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = _validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        code = _get_param(query_params, "code")
        if not code:
            return self._redirect_with_error("Missing authorization code")

        try:
            token_data = self._exchange_microsoft_code(code)
        except Exception as e:
            logger.error(f"Microsoft token exchange failed: {e}")
            return self._redirect_with_error("Failed to exchange authorization code")

        access_token = token_data.get("access_token")
        if not access_token:
            return self._redirect_with_error("No access token received")

        try:
            user_info = self._get_microsoft_user_info(access_token)
        except Exception as e:
            logger.error(f"Failed to get Microsoft user info: {e}")
            return self._redirect_with_error("Failed to get user info")

        return self._complete_oauth_flow(user_info, state_data)

    def _exchange_microsoft_code(self, code: str) -> dict:
        """Exchange Microsoft authorization code for access token."""
        import urllib.request

        tenant = _get_microsoft_tenant()
        token_url = MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant=tenant)

        data = urlencode(
            {
                "code": code,
                "client_id": _get_microsoft_client_id(),
                "client_secret": _get_microsoft_client_secret(),
                "redirect_uri": _get_microsoft_redirect_uri(),
                "grant_type": "authorization_code",
            }
        ).encode()

        req = urllib.request.Request(
            token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    def _get_microsoft_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user info from Microsoft Graph API."""
        import urllib.request

        req = urllib.request.Request(
            MICROSOFT_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            user_data = json.loads(response.read().decode())

        email = user_data.get("mail") or user_data.get("userPrincipalName", "")
        if not email or "@" not in email:
            raise ValueError("Could not retrieve email from Microsoft")

        return OAuthUserInfo(
            provider="microsoft",
            provider_user_id=user_data["id"],
            email=email,
            name=user_data.get("displayName", email.split("@")[0]),
            picture=None,  # Microsoft Graph requires separate call for photo
            email_verified=True,  # Microsoft validates emails
        )
