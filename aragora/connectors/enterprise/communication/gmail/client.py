"""
Gmail OAuth2 client and API request handling.

Provides authentication flow, token management, and base API request
infrastructure for the Gmail connector.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol

from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

class EnterpriseConnectorMethods(Protocol):
    """Protocol defining expected methods from EnterpriseConnector base class."""

    def check_circuit_breaker(self) -> bool: ...
    def get_circuit_breaker_status(self) -> dict[str, Any]: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...

# Gmail API scopes
# Note: gmail.metadata doesn't support search queries ('q' parameter)
# Using gmail.readonly alone is sufficient for read operations including search
GMAIL_SCOPES_READONLY = [
    "https://www.googleapis.com/auth/gmail.readonly",
]

# Full scopes including send (required for bidirectional email)
GMAIL_SCOPES_FULL = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.metadata",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Default to read-only for backward compatibility
GMAIL_SCOPES = GMAIL_SCOPES_READONLY

def _get_client_credentials() -> tuple[str, str]:
    """Get OAuth client ID and secret from environment."""
    client_id = (
        os.environ.get("GMAIL_CLIENT_ID")
        or os.environ.get("GOOGLE_GMAIL_CLIENT_ID")
        or os.environ.get("GOOGLE_CLIENT_ID", "")
    )
    client_secret = (
        os.environ.get("GMAIL_CLIENT_SECRET")
        or os.environ.get("GOOGLE_GMAIL_CLIENT_SECRET")
        or os.environ.get("GOOGLE_CLIENT_SECRET", "")
    )
    return client_id, client_secret

class GmailClientMixin(EnterpriseConnectorMethods):
    """Mixin providing OAuth2 authentication and API request infrastructure."""

    # These attributes are expected to be set by the concrete class
    _access_token: str | None
    _refresh_token: str | None
    _token_expiry: datetime | None
    _token_lock: asyncio.Lock
    user_id: str

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Gmail"

    @property
    def access_token(self) -> str | None:
        """Expose current access token (if available)."""
        return self._access_token

    @property
    def refresh_token(self) -> str | None:
        """Expose current refresh token (if available)."""
        return self._refresh_token

    @property
    def token_expiry(self) -> datetime | None:
        """Expose access token expiry (if available)."""
        return self._token_expiry

    @property
    def is_configured(self) -> bool:
        """Check if connector has required configuration."""
        return bool(
            os.environ.get("GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_CLIENT_ID")
        )

    def get_oauth_url(self, redirect_uri: str, state: str = "") -> str:
        """
        Generate OAuth2 authorization URL.

        Args:
            redirect_uri: URL to redirect after authorization
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL for user to visit
        """
        from urllib.parse import urlencode

        client_id, _ = _get_client_credentials()

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GMAIL_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }

        if state:
            params["state"] = state

        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    async def authenticate(
        self,
        code: str | None = None,
        redirect_uri: str | None = None,
        refresh_token: str | None = None,
    ) -> bool:
        """
        Authenticate with Gmail API.

        Either exchange authorization code for tokens, or use existing refresh token.

        Args:
            code: Authorization code from OAuth callback
            redirect_uri: Redirect URI used in authorization
            refresh_token: Existing refresh token

        Returns:
            True if authentication successful
        """
        import httpx

        client_id, client_secret = _get_client_credentials()

        if not client_id or not client_secret:
            logger.error("[Gmail] Missing OAuth credentials")
            return False

        try:
            async with httpx.AsyncClient() as client:
                if code and redirect_uri:
                    # Exchange code for tokens
                    response = await client.post(
                        "https://oauth2.googleapis.com/token",
                        data={
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "code": code,
                            "redirect_uri": redirect_uri,
                            "grant_type": "authorization_code",
                        },
                    )
                elif refresh_token:
                    # Use refresh token
                    response = await client.post(
                        "https://oauth2.googleapis.com/token",
                        data={
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "refresh_token": refresh_token,
                            "grant_type": "refresh_token",
                        },
                    )
                else:
                    logger.error("[Gmail] No code or refresh_token provided")
                    return False

                response.raise_for_status()
                data = response.json()

            self._access_token = data["access_token"]
            self._refresh_token = data.get("refresh_token", refresh_token)

            expires_in = data.get("expires_in", 3600)
            self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)

            logger.info("[Gmail] Authentication successful")
            return True

        except Exception as e:
            logger.error(f"[Gmail] Authentication failed: {e}")
            return False

    async def _refresh_access_token(self) -> str:
        """Refresh the access token using refresh token."""
        import httpx

        if not self._refresh_token:
            raise ValueError("No refresh token available")

        client_id, client_secret = _get_client_credentials()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": self._refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            data = response.json()

        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)

        return self._access_token

    async def _get_access_token(self) -> str:
        """Get valid access token, refreshing if needed.

        Thread-safe: Uses _token_lock to prevent concurrent refresh attempts.
        """
        async with self._token_lock:
            now = datetime.now(timezone.utc)

            if self._access_token and self._token_expiry and now < self._token_expiry:
                return self._access_token

            if self._refresh_token:
                return await self._refresh_access_token()

            raise ValueError("No valid access token and no refresh token available")

    async def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make a request to Gmail API with circuit breaker protection."""
        import httpx

        # Check circuit breaker first
        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        url = f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}{endpoint}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=60,
                )
                if response.status_code >= 400:
                    # Log the full error response for debugging
                    logger.error(f"Gmail API error {response.status_code}: {response.text}")
                    # Record failure for circuit breaker on 5xx errors or rate limits
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                response.raise_for_status()
                self.record_success()
                return response.json() if response.content else {}
        except httpx.TimeoutException as e:
            self.record_failure()
            logger.error(f"Gmail API timeout: {e}")
            raise
        except httpx.HTTPStatusError:
            # Already handled above
            raise
        except Exception as e:
            self.record_failure()
            logger.error(f"Gmail API error: {e}")
            raise

    def _get_client(self):
        """Get HTTP client context manager for API requests."""
        import httpx

        return httpx.AsyncClient(timeout=60)

    async def get_user_info(self) -> dict[str, Any]:
        """Get authenticated user's Gmail profile."""
        return await self._api_request("/profile")
