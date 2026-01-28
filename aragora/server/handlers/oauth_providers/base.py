"""
Base OAuth Provider - Abstract base class for OAuth providers.

Provides common infrastructure for OAuth 2.0 / OpenID Connect flows:
- Authorization URL generation
- Token exchange
- User info retrieval
- Token refresh

Subclasses implement provider-specific logic.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)


def _get_secret(name: str, default: str = "") -> str:
    """Get a secret from AWS Secrets Manager or environment."""
    try:
        from aragora.config.secrets import get_secret

        return get_secret(name, default) or default
    except ImportError:
        return os.environ.get(name, default)


def _is_production() -> bool:
    """Check if we're in production mode."""
    return os.environ.get("ARAGORA_ENV", "").lower() == "production"


@dataclass
class OAuthProviderConfig:
    """Configuration for an OAuth provider."""

    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: List[str] = field(default_factory=list)
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    userinfo_endpoint: str = ""
    revocation_endpoint: str = ""

    # Provider-specific settings
    tenant: Optional[str] = None  # For Microsoft
    team_id: Optional[str] = None  # For Apple
    key_id: Optional[str] = None  # For Apple
    private_key: Optional[str] = None  # For Apple


@dataclass
class OAuthTokens:
    """OAuth token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    id_token: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuthTokens":
        """Create from token endpoint response."""
        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in"),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
            id_token=data.get("id_token"),
        )


@dataclass
class OAuthUserInfo:
    """User information from OAuth provider."""

    provider: str
    provider_user_id: str
    email: Optional[str] = None
    email_verified: bool = False
    name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture: Optional[str] = None
    locale: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


class OAuthProvider(ABC):
    """
    Abstract base class for OAuth providers.

    Subclasses must implement:
    - PROVIDER_NAME: str - Provider identifier
    - get_authorization_url() - Generate auth URL
    - exchange_code() - Exchange code for tokens
    - get_user_info() - Get user info from tokens
    """

    PROVIDER_NAME: str = "unknown"

    def __init__(self, config: Optional[OAuthProviderConfig] = None):
        """
        Initialize the provider.

        Args:
            config: Optional explicit configuration. If not provided,
                    configuration is loaded from environment.
        """
        self._config = config or self._load_config_from_env()
        self._http_client: Optional[httpx.Client] = None

    @abstractmethod
    def _load_config_from_env(self) -> OAuthProviderConfig:
        """Load configuration from environment variables."""
        ...

    @property
    def config(self) -> OAuthProviderConfig:
        """Get the provider configuration."""
        return self._config

    @property
    def is_configured(self) -> bool:
        """Check if the provider has required configuration."""
        return bool(self._config.client_id and self._config.client_secret)

    def _get_http_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)
        return self._http_client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> "OAuthProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # =========================================================================
    # OAuth Flow Methods
    # =========================================================================

    @abstractmethod
    def get_authorization_url(
        self,
        state: str,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate the authorization URL for OAuth consent.

        Args:
            state: CSRF protection state parameter
            redirect_uri: Override redirect URI
            scopes: Override scopes
            **kwargs: Provider-specific parameters

        Returns:
            Authorization URL to redirect user to
        """
        ...

    @abstractmethod
    def exchange_code(
        self,
        code: str,
        redirect_uri: Optional[str] = None,
    ) -> OAuthTokens:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            redirect_uri: Redirect URI used in authorization

        Returns:
            OAuth tokens
        """
        ...

    @abstractmethod
    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from the provider.

        Args:
            access_token: Access token from exchange

        Returns:
            User information
        """
        ...

    def exchange_and_get_user(
        self,
        code: str,
        redirect_uri: Optional[str] = None,
    ) -> OAuthUserInfo:
        """
        Exchange code and get user info in one call.

        Args:
            code: Authorization code
            redirect_uri: Redirect URI

        Returns:
            User information
        """
        tokens = self.exchange_code(code, redirect_uri)
        return self.get_user_info(tokens.access_token)

    def refresh_access_token(self, refresh_token: str) -> OAuthTokens:
        """
        Refresh the access token.

        Args:
            refresh_token: Refresh token from previous exchange

        Returns:
            New OAuth tokens

        Raises:
            NotImplementedError: If provider doesn't support refresh
        """
        raise NotImplementedError(f"{self.PROVIDER_NAME} does not support token refresh")

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token.

        Args:
            token: Token to revoke (access or refresh)

        Returns:
            True if revocation succeeded
        """
        if not self._config.revocation_endpoint:
            return False

        try:
            client = self._get_http_client()
            response = client.post(
                self._config.revocation_endpoint,
                data={"token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"[{self.PROVIDER_NAME}] Token revocation failed: {e}")
            return False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_authorization_url(
        self,
        endpoint: str,
        params: Dict[str, str],
    ) -> str:
        """Build authorization URL with query parameters."""
        return f"{endpoint}?{urlencode(params)}"

    def _request_tokens(
        self,
        endpoint: str,
        data: Dict[str, str],
        headers: Optional[Dict[str, str]] = None,
    ) -> OAuthTokens:
        """Make token request to provider."""
        client = self._get_http_client()
        request_headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if headers:
            request_headers.update(headers)

        response = client.post(endpoint, data=data, headers=request_headers)
        response.raise_for_status()

        return OAuthTokens.from_dict(response.json())

    def _request_user_info(
        self,
        endpoint: str,
        access_token: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make user info request to provider."""
        client = self._get_http_client()
        request_headers = {"Authorization": f"Bearer {access_token}"}
        if headers:
            request_headers.update(headers)

        response = client.get(endpoint, headers=request_headers)
        response.raise_for_status()

        return response.json()


__all__ = [
    "OAuthProvider",
    "OAuthProviderConfig",
    "OAuthTokens",
    "OAuthUserInfo",
    "_get_secret",
    "_is_production",
]
