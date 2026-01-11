"""
SSO Authentication Handler for Aragora.

Provides endpoints for enterprise SSO authentication:
- /auth/sso/login - Initiate SSO login
- /auth/sso/callback - Handle IdP callback
- /auth/sso/logout - Handle logout
- /auth/sso/metadata - SAML SP metadata (SAML only)

Usage:
    from aragora.server.handlers.sso import SSOHandler

    # Register with unified server
    server.add_handler(SSOHandler())
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from .base import BaseHandler, HandlerResult, json_response, error_response

logger = logging.getLogger(__name__)


class SSOHandler(BaseHandler):
    """
    Handler for SSO authentication endpoints.

    Supports SAML 2.0 and OpenID Connect (OIDC) providers.
    """

    def __init__(self):
        """Initialize SSO handler."""
        self._provider = None
        self._initialized = False

    def _get_provider(self):
        """Lazy-load SSO provider."""
        if not self._initialized:
            try:
                from aragora.auth import get_sso_provider
                self._provider = get_sso_provider()
            except ImportError:
                logger.warning("SSO auth module not available")
            self._initialized = True
        return self._provider

    def routes(self) -> list[tuple[str, str, str]]:
        """Return SSO routes."""
        return [
            ("GET", "/auth/sso/login", "handle_login"),
            ("POST", "/auth/sso/login", "handle_login"),
            ("GET", "/auth/sso/callback", "handle_callback"),
            ("POST", "/auth/sso/callback", "handle_callback"),
            ("GET", "/auth/sso/logout", "handle_logout"),
            ("POST", "/auth/sso/logout", "handle_logout"),
            ("GET", "/auth/sso/metadata", "handle_metadata"),
            ("GET", "/auth/sso/status", "handle_status"),
        ]

    async def handle_login(self, handler, params: Dict[str, Any]) -> HandlerResult:
        """
        Initiate SSO login flow.

        GET/POST /auth/sso/login

        Query params:
            - redirect_uri: Optional redirect after login
            - state: Optional state parameter

        Returns:
            Redirect to IdP or JSON with auth URL
        """
        provider = self._get_provider()
        if not provider:
            return error_response(
                "SSO not configured",
                501,
                code="SSO_NOT_CONFIGURED",
                suggestion="Configure SSO in environment variables (ARAGORA_SSO_*)"
            )

        try:
            # Get parameters
            redirect_uri = params.get("redirect_uri", [""])[0] if isinstance(params.get("redirect_uri"), list) else params.get("redirect_uri", "")
            state = params.get("state", [""])[0] if isinstance(params.get("state"), list) else params.get("state", "")

            # Generate state if not provided
            if not state:
                state = provider.generate_state()

            # Get authorization URL
            auth_url = await provider.get_authorization_url(
                state=state,
                relay_state=redirect_uri or None,
            )

            # Check if client wants JSON or redirect
            accept = handler.headers.get("Accept", "") if hasattr(handler, "headers") else ""
            if "application/json" in accept:
                return json_response({
                    "auth_url": auth_url,
                    "state": state,
                    "provider": provider.provider_type.value,
                })
            else:
                # Return redirect response
                return {
                    "status": 302,
                    "headers": {
                        "Location": auth_url,
                        "Cache-Control": "no-cache, no-store",
                    },
                    "body": "",
                }

        except Exception as e:
            logger.error(f"SSO login error: {e}")
            return error_response(
                f"SSO login failed: {e}",
                500,
                code="SSO_LOGIN_ERROR"
            )

    async def handle_callback(self, handler, params: Dict[str, Any]) -> HandlerResult:
        """
        Handle IdP callback after authentication.

        GET/POST /auth/sso/callback

        For OIDC:
            - code: Authorization code
            - state: State parameter

        For SAML:
            - SAMLResponse: Base64-encoded SAML response
            - RelayState: Original state

        Returns:
            JWT session token and user info
        """
        provider = self._get_provider()
        if not provider:
            return error_response(
                "SSO not configured",
                501,
                code="SSO_NOT_CONFIGURED"
            )

        try:
            # Extract callback parameters
            code = self._get_param(params, "code")
            state = self._get_param(params, "state")
            saml_response = self._get_param(params, "SAMLResponse")
            relay_state = self._get_param(params, "RelayState") or state

            # Check for error from IdP
            error = self._get_param(params, "error")
            if error:
                error_desc = self._get_param(params, "error_description") or error
                return error_response(
                    f"IdP error: {error_desc}",
                    401,
                    code="SSO_IDP_ERROR"
                )

            # Authenticate
            user = await provider.authenticate(
                code=code,
                saml_response=saml_response,
                state=relay_state,
            )

            # Generate session token
            from aragora.server.auth import auth_config
            session_token = auth_config.generate_token(
                loop_id=user.id,
                expires_in=provider.config.session_duration_seconds,
            )

            # Return user info with token
            response_data = {
                "success": True,
                "user": user.to_dict(),
                "token": session_token,
                "expires_in": provider.config.session_duration_seconds,
            }

            # Check if we should redirect
            if relay_state and relay_state.startswith(("http://", "https://")):
                # Redirect with token
                separator = "&" if "?" in relay_state else "?"
                redirect_url = f"{relay_state}{separator}token={session_token}"
                return {
                    "status": 302,
                    "headers": {
                        "Location": redirect_url,
                        "Cache-Control": "no-cache, no-store",
                    },
                    "body": "",
                }

            return json_response(response_data)

        except Exception as e:
            logger.error(f"SSO callback error: {e}")

            # Handle specific errors
            error_msg = str(e)
            if "DOMAIN_NOT_ALLOWED" in error_msg:
                return error_response(
                    error_msg,
                    403,
                    code="SSO_DOMAIN_NOT_ALLOWED",
                    suggestion="Contact your administrator to add your domain"
                )
            elif "INVALID_STATE" in error_msg:
                return error_response(
                    "Session expired. Please try logging in again.",
                    401,
                    code="SSO_SESSION_EXPIRED",
                    suggestion="Click the login button to start a new session"
                )

            return error_response(
                f"Authentication failed: {e}",
                401,
                code="SSO_AUTH_FAILED"
            )

    async def handle_logout(self, handler, params: Dict[str, Any]) -> HandlerResult:
        """
        Handle SSO logout.

        GET/POST /auth/sso/logout

        Returns:
            Redirect to IdP logout or success message
        """
        provider = self._get_provider()
        if not provider:
            return json_response({"success": True, "message": "Logged out"})

        try:
            # Get current user token
            from aragora.server.auth import auth_config

            token = None
            if hasattr(handler, "headers"):
                auth_header = handler.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]

            # Revoke token
            if token:
                auth_config.revoke_token(token, "user_logout")

            # Get IdP logout URL
            from aragora.auth.sso import SSOUser
            logout_url = await provider.logout(SSOUser(id="", email=""))

            if logout_url:
                return {
                    "status": 302,
                    "headers": {
                        "Location": logout_url,
                        "Cache-Control": "no-cache, no-store",
                    },
                    "body": "",
                }

            return json_response({
                "success": True,
                "message": "Logged out successfully",
            })

        except Exception as e:
            logger.error(f"SSO logout error: {e}")
            return json_response({
                "success": True,
                "message": "Logged out (with errors)",
            })

    async def handle_metadata(self, handler, params: Dict[str, Any]) -> HandlerResult:
        """
        Get SAML SP metadata.

        GET /auth/sso/metadata

        Returns:
            XML metadata document for SAML providers
        """
        provider = self._get_provider()
        if not provider:
            return error_response(
                "SSO not configured",
                501,
                code="SSO_NOT_CONFIGURED"
            )

        # Check if SAML provider
        from aragora.auth.sso import SSOProviderType
        if provider.provider_type != SSOProviderType.SAML:
            return error_response(
                "Metadata only available for SAML providers",
                400,
                code="NOT_SAML_PROVIDER"
            )

        try:
            from aragora.auth.saml import SAMLProvider
            if isinstance(provider, SAMLProvider):
                metadata = await provider.get_metadata()
                return {
                    "status": 200,
                    "headers": {
                        "Content-Type": "application/xml",
                        "Cache-Control": "max-age=3600",
                    },
                    "body": metadata,
                }

        except Exception as e:
            logger.error(f"Metadata generation error: {e}")
            return error_response(
                f"Failed to generate metadata: {e}",
                500,
                code="METADATA_ERROR"
            )

        return error_response("Metadata not available", 400)

    async def handle_status(self, handler, params: Dict[str, Any]) -> HandlerResult:
        """
        Get SSO configuration status.

        GET /auth/sso/status

        Returns:
            SSO configuration status and provider info
        """
        provider = self._get_provider()

        if not provider:
            return json_response({
                "enabled": False,
                "configured": False,
                "provider": None,
                "message": "SSO not configured",
            })

        return json_response({
            "enabled": True,
            "configured": True,
            "provider": provider.provider_type.value,
            "entity_id": provider.config.entity_id,
            "callback_url": provider.config.callback_url,
            "auto_provision": provider.config.auto_provision,
            "allowed_domains": provider.config.allowed_domains if hasattr(provider.config, "allowed_domains") else [],
        })

    def _get_param(self, params: Dict[str, Any], key: str) -> Optional[str]:
        """Extract parameter value, handling list format."""
        value = params.get(key)
        if value is None:
            return None
        if isinstance(value, list):
            return value[0] if value else None
        return str(value)


__all__ = ["SSOHandler"]
