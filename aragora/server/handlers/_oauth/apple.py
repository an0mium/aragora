"""
Apple OAuth mixin.

Provides Apple Sign-In authentication methods for the OAuthHandler.
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


class AppleOAuthMixin:
    """Mixin providing Apple Sign-In methods."""

    @handle_errors("Apple OAuth start")
    @log_request("Apple OAuth start")
    def _handle_apple_auth_start(self, handler, query_params: dict) -> HandlerResult:
        """Redirect user to Apple OAuth consent screen."""
        impl = _impl()
        apple_client_id = impl._get_apple_client_id()
        if not apple_client_id:
            return error_response("Apple OAuth not configured", 503)

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

        params = {
            "client_id": apple_client_id,
            "redirect_uri": impl._get_apple_redirect_uri(),
            "response_type": "code id_token",
            "scope": "name email",
            "state": state,
            "response_mode": "form_post",  # Apple requires form_post
        }
        auth_url = f"{impl.APPLE_AUTH_URL}?{urlencode(params)}"

        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=f'<html><head><meta http-equiv="refresh" content="0;url={auth_url}"></head></html>'.encode(),
            headers={"Location": auth_url},
        )

    @handle_errors("Apple OAuth callback")
    @log_request("Apple OAuth callback")
    def _handle_apple_callback(self, handler, query_params: dict) -> HandlerResult:
        """Handle Apple OAuth callback (POST with form data)."""
        impl = _impl()

        # Apple uses form_post, so we need to read POST body
        body_data: dict[str, str] = {}
        if hasattr(handler, "request") and handler.request.body:
            from urllib.parse import parse_qs

            parsed = parse_qs(handler.request.body.decode())
            body_data = {k: v[0] if v else "" for k, v in parsed.items()}

        # Merge with query params (for GET fallback)
        all_params = {**query_params, **body_data}

        error = all_params.get("error")
        if error:
            return self._redirect_with_error(f"Apple OAuth error: {error}")

        state = all_params.get("state")
        if not state:
            return self._redirect_with_error("Missing state parameter")

        state_data = impl._validate_state(state)
        if state_data is None:
            return self._redirect_with_error("Invalid or expired state")

        code = all_params.get("code")
        id_token = all_params.get("id_token")

        if not code and not id_token:
            return self._redirect_with_error("Missing authorization code or id_token")

        # Apple provides user info only on first authorization
        user_data_str = all_params.get("user", "{}")
        try:
            user_data = json.loads(user_data_str) if user_data_str else {}
        except json.JSONDecodeError:
            user_data = {}

        try:
            if code:
                token_data = self._exchange_apple_code(code)
                id_token = token_data.get("id_token", id_token)

            user_info = self._parse_apple_id_token(id_token, user_data)
        except Exception as e:
            logger.error(f"Apple OAuth processing failed: {e}")
            return self._redirect_with_error("Failed to process Apple sign-in")

        return self._complete_oauth_flow(user_info, state_data)

    def _exchange_apple_code(self, code: str) -> dict:
        """Exchange Apple authorization code for tokens."""
        import urllib.request

        impl = _impl()
        client_secret = self._generate_apple_client_secret()

        data = urlencode(
            {
                "code": code,
                "client_id": impl._get_apple_client_id(),
                "client_secret": client_secret,
                "redirect_uri": impl._get_apple_redirect_uri(),
                "grant_type": "authorization_code",
            }
        ).encode()

        req = urllib.request.Request(
            impl.APPLE_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    def _generate_apple_client_secret(self) -> str:
        """Generate Apple client secret JWT."""
        try:
            import jwt
        except ImportError:
            raise ValueError("PyJWT required for Apple OAuth. Install with: pip install PyJWT")

        impl = _impl()
        team_id = impl._get_apple_team_id()
        key_id = impl._get_apple_key_id()
        private_key = impl._get_apple_private_key()
        client_id = impl._get_apple_client_id()

        if not all([team_id, key_id, private_key, client_id]):
            raise ValueError("Apple OAuth not fully configured")

        now = int(time.time())
        payload = {
            "iss": team_id,
            "iat": now,
            "exp": now + 86400 * 180,  # 180 days max
            "aud": "https://appleid.apple.com",
            "sub": client_id,
        }

        return jwt.encode(payload, private_key, algorithm="ES256", headers={"kid": key_id})

    def _parse_apple_id_token(self, id_token: str, user_data: dict) -> OAuthUserInfo:
        """Parse Apple ID token to extract user info."""
        import base64

        # Decode JWT payload (Apple signs it, but we trust it from their endpoint)
        parts = id_token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid Apple ID token format")

        # Decode payload with padding
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        email = payload.get("email", "")
        email_verified = payload.get("email_verified", False)
        sub = payload.get("sub", "")

        if not email:
            raise ValueError("No email in Apple ID token")

        # Apple only sends name on first auth, stored in user_data
        name_data = user_data.get("name", {})
        name = ""
        if name_data:
            first = name_data.get("firstName", "")
            last = name_data.get("lastName", "")
            name = f"{first} {last}".strip()
        if not name:
            name = email.split("@")[0]

        return OAuthUserInfo(
            provider="apple",
            provider_user_id=sub,
            email=email,
            name=name,
            picture=None,  # Apple doesn't provide profile pictures
            email_verified=email_verified == "true" or email_verified is True,
        )
