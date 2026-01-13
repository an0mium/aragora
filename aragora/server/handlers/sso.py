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
import os
from typing import Any, Dict, Optional, Union

from .base import BaseHandler, HandlerResult, json_response, error_response, handle_errors
from .utils.rate_limit import rate_limit
from aragora.exceptions import ConfigurationError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


try:
    from aragora.auth import get_sso_provider as _get_sso_provider
except ImportError:  # pragma: no cover - optional dependency

    def _get_sso_provider():
        raise ImportError("SSO auth module not available")


get_sso_provider = _get_sso_provider

try:
    from aragora.server.auth import auth_config as auth_config
except ImportError:  # pragma: no cover - optional dependency
    auth_config = None

try:
    from aragora.auth.sso import SSOUser, SSOProviderType
except ImportError:  # pragma: no cover - optional dependency
    SSOUser = None
    SSOProviderType = None

try:
    from aragora.auth.saml import SAMLProvider
except ImportError:  # pragma: no cover - optional dependency
    SAMLProvider = None


class SSOHandler(BaseHandler):
    """
    Handler for SSO authentication endpoints.

    Supports SAML 2.0 and OpenID Connect (OIDC) providers.
    """

    def __init__(self, server_context: Optional[dict] = None):
        """Initialize SSO handler."""
        super().__init__(server_context or {})
        self._provider = None
        self._initialized = False

    def _get_provider(self):
        """Lazy-load SSO provider."""
        if not self._initialized:
            try:
                self._provider = get_sso_provider()
            except ImportError:
                logger.warning("SSO auth module not available")
            self._initialized = True
        return self._provider

    def _should_return_handler_result(self, handler: Any) -> bool:
        """Determine whether to return HandlerResult or legacy dict response."""
        if handler is None:
            return False
        # Avoid isinstance to keep tests that patch it from breaking.
        return "send_response" in dir(handler)

    def _flatten_error_body(self, body: Any) -> Any:
        """Convert structured error payloads to legacy flat shape."""
        if not isinstance(body, dict):
            return body
        error = body.get("error")
        if not isinstance(error, dict):
            return body

        flat = {k: v for k, v in body.items() if k != "error"}
        flat["error"] = error.get("message", error)
        for key in ("code", "suggestion", "details", "trace_id"):
            if key in error:
                flat[key] = error[key]
        return flat

    def _to_legacy_result(self, result: Union[HandlerResult, dict]) -> dict:
        """Normalize HandlerResult to dict for legacy/tests."""
        if isinstance(result, dict):
            legacy = dict(result)
            body = legacy.get("body", {})
            content_type = legacy.get("content_type") or legacy.get("Content-Type")
            if (
                isinstance(body, (bytes, str))
                and content_type
                and str(content_type).startswith("application/json")
            ):
                try:
                    if isinstance(body, bytes):
                        body = body.decode("utf-8")
                    body = json.loads(body)
                except (ValueError, TypeError):
                    pass
            elif isinstance(body, bytes):
                body = body.decode("utf-8", errors="replace")
            legacy["body"] = self._flatten_error_body(body)
            legacy.setdefault("headers", {})
            if "status" not in legacy and "status_code" in legacy:
                legacy["status"] = legacy["status_code"]
            return legacy

        result_body: Any = result.body
        if result.content_type and result.content_type.startswith("application/json"):
            try:
                result_body = json.loads(result.body.decode("utf-8"))
            except (ValueError, TypeError):
                result_body = result.body.decode("utf-8", errors="replace")
        elif isinstance(result_body, bytes):
            result_body = result_body.decode("utf-8", errors="replace")

        return {
            "status": result.status_code,
            "headers": result.headers or {},
            "body": self._flatten_error_body(result_body),
        }

    def _format_response(
        self, handler: Any, result: Union[HandlerResult, dict]
    ) -> Union[HandlerResult, dict]:
        """Return handler result or legacy dict depending on context."""
        if self._should_return_handler_result(handler):
            return result
        return self._to_legacy_result(result)

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

    async def handle_login(self, handler: Any, params: Dict[str, Any]) -> Union[HandlerResult, Dict[str, Any]]:
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
            return self._format_response(
                handler,
                error_response(
                    "SSO not configured",
                    501,
                    code="SSO_NOT_CONFIGURED",
                    suggestion="Configure SSO in environment variables (ARAGORA_SSO_*)",
                ),
            )

        try:
            # Get parameters
            redirect_uri = (
                params.get("redirect_uri", [""])[0]
                if isinstance(params.get("redirect_uri"), list)
                else params.get("redirect_uri", "")
            )
            state = (
                params.get("state", [""])[0]
                if isinstance(params.get("state"), list)
                else params.get("state", "")
            )

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
                return self._format_response(
                    handler,
                    json_response(
                        {
                            "auth_url": auth_url,
                            "state": state,
                            "provider": provider.provider_type.value,
                        }
                    ),
                )
            else:
                # Return redirect response
                return self._format_response(
                    handler,
                    HandlerResult(
                        status_code=302,
                        content_type="text/plain",
                        body=b"",
                        headers={
                            "Location": auth_url,
                            "Cache-Control": "no-cache, no-store",
                        },
                    ),
                )

        except Exception as e:
            logger.error(f"SSO login error: {e}")
            return self._format_response(
                handler, error_response(f"SSO login failed: {e}", 500, code="SSO_LOGIN_ERROR")
            )

    async def handle_callback(self, handler: Any, params: Dict[str, Any]) -> Union[HandlerResult, Dict[str, Any]]:
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
            return self._format_response(
                handler, error_response("SSO not configured", 501, code="SSO_NOT_CONFIGURED")
            )

        # SECURITY: Enforce HTTPS for callbacks in production
        if os.getenv("ARAGORA_ENV") == "production":
            callback_url = provider.config.callback_url
            if callback_url and not callback_url.startswith("https://"):
                logger.error(f"SSO callback URL must use HTTPS in production: {callback_url}")
                return self._format_response(
                    handler,
                    error_response(
                        "SSO callback URL must use HTTPS in production",
                        400,
                        code="INSECURE_CALLBACK_URL",
                        suggestion="Configure ARAGORA_SSO_CALLBACK_URL with https://",
                    ),
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
                return self._format_response(
                    handler, error_response(f"IdP error: {error_desc}", 401, code="SSO_IDP_ERROR")
                )

            # Authenticate
            user = await provider.authenticate(
                code=code,
                saml_response=saml_response,
                state=relay_state,
            )

            # Generate session token
            if not auth_config:
                raise ConfigurationError("SSOHandler", "auth_config not initialized")
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
                # SECURITY: Validate redirect URL before redirecting
                if not self._validate_redirect_url(relay_state):
                    logger.warning(f"SSO callback: blocked unsafe redirect to {relay_state}")
                    return self._format_response(
                        handler,
                        error_response(
                            "Invalid redirect URL",
                            400,
                            code="SSO_INVALID_REDIRECT",
                            suggestion="Redirect URL must be on the allowed hosts list",
                        ),
                    )

                # Redirect with token
                separator = "&" if "?" in relay_state else "?"
                redirect_url = f"{relay_state}{separator}token={session_token}"
                return self._format_response(
                    handler,
                    HandlerResult(
                        status_code=302,
                        content_type="text/plain",
                        body=b"",
                        headers={
                            "Location": redirect_url,
                            "Cache-Control": "no-cache, no-store",
                        },
                    ),
                )

            return self._format_response(handler, json_response(response_data))

        except Exception as e:
            logger.error(f"SSO callback error: {e}")

            # Handle specific errors
            error_msg = str(e)
            if "DOMAIN_NOT_ALLOWED" in error_msg:
                return self._format_response(
                    handler,
                    error_response(
                        error_msg,
                        403,
                        code="SSO_DOMAIN_NOT_ALLOWED",
                        suggestion="Contact your administrator to add your domain",
                    ),
                )
            elif "INVALID_STATE" in error_msg:
                return self._format_response(
                    handler,
                    error_response(
                        "Session expired. Please try logging in again.",
                        401,
                        code="SSO_SESSION_EXPIRED",
                        suggestion="Click the login button to start a new session",
                    ),
                )

            return self._format_response(
                handler, error_response(f"Authentication failed: {e}", 401, code="SSO_AUTH_FAILED")
            )

    async def handle_logout(self, handler: Any, params: Dict[str, Any]) -> Union[HandlerResult, Dict[str, Any]]:
        """
        Handle SSO logout.

        GET/POST /auth/sso/logout

        Returns:
            Redirect to IdP logout or success message
        """
        provider = self._get_provider()
        if not provider:
            return self._format_response(
                handler, json_response({"success": True, "message": "Logged out"})
            )

        try:
            # Get current user token
            if not auth_config:
                raise ConfigurationError("SSOHandler", "auth_config not initialized")

            token = None
            if hasattr(handler, "headers"):
                auth_header = handler.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]

            # Revoke token
            if token:
                auth_config.revoke_token(token, "user_logout")

            # Get IdP logout URL
            if not SSOUser:
                raise ConfigurationError("SSOHandler", "SSOUser type not imported")
            logout_url = await provider.logout(SSOUser(id="", email=""))

            if logout_url:
                return self._format_response(
                    handler,
                    HandlerResult(
                        status_code=302,
                        content_type="text/plain",
                        body=b"",
                        headers={
                            "Location": logout_url,
                            "Cache-Control": "no-cache, no-store",
                        },
                    ),
                )

            return self._format_response(
                handler,
                json_response(
                    {
                        "success": True,
                        "message": "Logged out successfully",
                    }
                ),
            )

        except Exception as e:
            logger.error(f"SSO logout error: {e}")
            return self._format_response(
                handler,
                json_response(
                    {
                        "success": True,
                        "message": "Logged out (with errors)",
                    }
                ),
            )

    async def handle_metadata(self, handler: Any, params: Dict[str, Any]) -> Union[HandlerResult, Dict[str, Any]]:
        """
        Get SAML SP metadata.

        GET /auth/sso/metadata

        Returns:
            XML metadata document for SAML providers
        """
        provider = self._get_provider()
        if not provider:
            return self._format_response(
                handler, error_response("SSO not configured", 501, code="SSO_NOT_CONFIGURED")
            )

        # Check if SAML provider
        if not SSOProviderType:
            return self._format_response(
                handler, error_response("SSO provider types unavailable", 503)
            )
        if provider.provider_type != SSOProviderType.SAML:
            return self._format_response(
                handler,
                error_response(
                    "Metadata only available for SAML providers", 400, code="NOT_SAML_PROVIDER"
                ),
            )

        try:
            if hasattr(provider, "get_metadata"):
                metadata = await provider.get_metadata()
                return self._format_response(
                    handler,
                    HandlerResult(
                        status_code=200,
                        content_type="application/xml",
                        body=str(metadata).encode("utf-8"),
                        headers={
                            "Content-Type": "application/xml",
                            "Cache-Control": "max-age=3600",
                        },
                    ),
                )

        except Exception as e:
            logger.error(f"Metadata generation error: {e}")
            return self._format_response(
                handler,
                error_response(f"Failed to generate metadata: {e}", 500, code="METADATA_ERROR"),
            )

        return self._format_response(handler, error_response("Metadata not available", 400))

    async def handle_status(self, handler: Any, params: Dict[str, Any]) -> Union[HandlerResult, Dict[str, Any]]:
        """
        Get SSO configuration status.

        GET /auth/sso/status

        Returns:
            SSO configuration status and provider info
        """
        provider = self._get_provider()

        if not provider:
            return self._format_response(
                handler,
                json_response(
                    {
                        "enabled": False,
                        "configured": False,
                        "provider": None,
                        "message": "SSO not configured",
                    }
                ),
            )

        return self._format_response(
            handler,
            json_response(
                {
                    "enabled": True,
                    "configured": True,
                    "provider": provider.provider_type.value,
                    "entity_id": provider.config.entity_id,
                    "callback_url": provider.config.callback_url,
                    "auto_provision": provider.config.auto_provision,
                    "allowed_domains": (
                        provider.config.allowed_domains
                        if hasattr(provider.config, "allowed_domains")
                        else []
                    ),
                }
            ),
        )

    def _get_param(self, params: Dict[str, Any], key: str) -> Optional[str]:
        """Extract parameter value, handling list format."""
        value = params.get(key)
        if value is None:
            return None
        if isinstance(value, list):
            return value[0] if value else None
        return str(value)

    def _validate_redirect_url(self, url: str) -> bool:
        """
        Validate that a redirect URL is safe.

        Prevents open redirect attacks by checking:
        1. URL uses allowed scheme (http/https)
        2. URL host is on the allowlist (if configured)
        3. URL doesn't contain dangerous patterns

        Returns True if URL is safe, False otherwise.
        """
        if not url:
            return True  # No redirect is safe

        try:
            parsed = urlparse(url)

            # Must be http or https
            if parsed.scheme not in ("http", "https"):
                logger.warning(f"SSO redirect blocked: invalid scheme {parsed.scheme}")
                return False

            # Check for dangerous patterns
            if "@" in parsed.netloc:  # user:pass@host trick
                logger.warning("SSO redirect blocked: credentials in URL")
                return False

            # Get allowed redirect hosts from environment
            allowed_hosts_str = os.getenv("ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS", "")
            if allowed_hosts_str:
                allowed_hosts = [h.strip().lower() for h in allowed_hosts_str.split(",")]
                host = parsed.netloc.lower().split(":")[0]  # Remove port

                if host not in allowed_hosts:
                    logger.warning(f"SSO redirect blocked: host {host} not in allowlist")
                    return False

            # In production, require HTTPS for redirects
            if os.getenv("ARAGORA_ENV") == "production" and parsed.scheme != "https":
                logger.warning("SSO redirect blocked: HTTPS required in production")
                return False

            return True

        except Exception as e:
            logger.warning(f"SSO redirect validation error: {e}")
            return False


__all__ = ["SSOHandler"]
