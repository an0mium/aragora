"""
OAuth Namespace API

Provides a namespaced interface for OAuth authentication flows.
Supports Google, GitHub, Microsoft, Apple, and generic OIDC providers.

Features:
- Getting authorization URLs for various providers
- Handling OAuth callbacks
- Linking/unlinking OAuth accounts
- Listing available and linked providers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


OAuthProvider = Literal["google", "github", "microsoft", "apple", "oidc"]


class OAuthAPI:
    """
    Synchronous OAuth API.

    Provides methods for OAuth authentication:
    - Getting authorization URLs for various providers
    - Handling OAuth callbacks
    - Linking/unlinking OAuth accounts
    - Listing available and linked providers

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> providers = client.oauth.get_providers()
        >>> auth_url = client.oauth.get_google_auth_url(redirect_uri="https://myapp.com/callback")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_providers(self) -> dict[str, Any]:
        """
        Get list of available OAuth providers.

        Returns:
            Dict with providers list containing:
            - provider: Provider name
            - enabled: Whether enabled
            - name: Display name
            - icon_url: Provider icon
        """
        return self._client.request("GET", "/api/v1/auth/oauth/providers")

    def get_linked_providers(self) -> dict[str, Any]:
        """
        Get user's linked OAuth providers.

        Returns:
            Dict with providers list containing linked accounts
        """
        return self._client.request("GET", "/api/v1/user/oauth-providers")

    def get_google_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """
        Get authorization URL for Google OAuth.

        Args:
            redirect_uri: URL to redirect after auth
            state: Optional state parameter for CSRF protection

        Returns:
            Dict with authorization_url and state
        """
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        return self._client.request(
            "GET", "/api/v1/auth/oauth/google", params=params if params else None
        )

    def get_github_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for GitHub OAuth."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        return self._client.request(
            "GET", "/api/v1/auth/oauth/github", params=params if params else None
        )

    def get_microsoft_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for Microsoft OAuth."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        return self._client.request(
            "GET", "/api/v1/auth/oauth/microsoft", params=params if params else None
        )

    def get_apple_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for Apple OAuth."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        return self._client.request(
            "GET", "/api/v1/auth/oauth/apple", params=params if params else None
        )

    def get_oidc_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
        provider_id: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for generic OIDC provider."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        if provider_id:
            params["provider_id"] = provider_id
        return self._client.request(
            "GET", "/api/v1/auth/oauth/oidc", params=params if params else None
        )

    def link_account(
        self,
        provider: OAuthProvider,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """
        Link an OAuth account to the current user.

        Args:
            provider: OAuth provider name
            code: Authorization code from provider
            redirect_uri: The redirect URI used in auth flow

        Returns:
            Dict with success status and linked account info
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/oauth/link",
            json={"provider": provider, "code": code, "redirect_uri": redirect_uri},
        )

    def unlink_account(self, provider: OAuthProvider) -> dict[str, Any]:
        """
        Unlink an OAuth provider from the current user.

        Args:
            provider: OAuth provider to unlink

        Returns:
            Dict with success status and message
        """
        return self._client.request(
            "DELETE",
            "/api/v1/auth/oauth/unlink",
            json={"provider": provider},
        )

    def handle_callback(
        self,
        provider: OAuthProvider,
        code: str,
        state: str | None = None,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        """
        Handle OAuth callback (exchange code for tokens).

        Args:
            provider: OAuth provider
            code: Authorization code
            state: State parameter
            redirect_uri: Redirect URI

        Returns:
            Dict with tokens and user info
        """
        path = (
            "/api/v1/auth/oauth/oidc/callback"
            if provider == "oidc"
            else f"/api/v1/auth/oauth/{provider}/callback"
        )
        params: dict[str, Any] = {"code": code}
        if state:
            params["state"] = state
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        return self._client.request("GET", path, params=params)


class AsyncOAuthAPI:
    """
    Asynchronous OAuth API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     providers = await client.oauth.get_providers()
        ...     auth_url = await client.oauth.get_google_auth_url()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_providers(self) -> dict[str, Any]:
        """Get list of available OAuth providers."""
        return await self._client.request("GET", "/api/v1/auth/oauth/providers")

    async def get_linked_providers(self) -> dict[str, Any]:
        """Get user's linked OAuth providers."""
        return await self._client.request("GET", "/api/v1/user/oauth-providers")

    async def get_google_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for Google OAuth."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        return await self._client.request(
            "GET", "/api/v1/auth/oauth/google", params=params if params else None
        )

    async def get_github_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for GitHub OAuth."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        return await self._client.request(
            "GET", "/api/v1/auth/oauth/github", params=params if params else None
        )

    async def get_microsoft_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for Microsoft OAuth."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        return await self._client.request(
            "GET", "/api/v1/auth/oauth/microsoft", params=params if params else None
        )

    async def get_apple_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for Apple OAuth."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        return await self._client.request(
            "GET", "/api/v1/auth/oauth/apple", params=params if params else None
        )

    async def get_oidc_auth_url(
        self,
        redirect_uri: str | None = None,
        state: str | None = None,
        provider_id: str | None = None,
    ) -> dict[str, Any]:
        """Get authorization URL for generic OIDC provider."""
        params: dict[str, Any] = {}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        if provider_id:
            params["provider_id"] = provider_id
        return await self._client.request(
            "GET", "/api/v1/auth/oauth/oidc", params=params if params else None
        )

    async def link_account(
        self,
        provider: OAuthProvider,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Link an OAuth account to the current user."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/oauth/link",
            json={"provider": provider, "code": code, "redirect_uri": redirect_uri},
        )

    async def unlink_account(self, provider: OAuthProvider) -> dict[str, Any]:
        """Unlink an OAuth provider from the current user."""
        return await self._client.request(
            "DELETE",
            "/api/v1/auth/oauth/unlink",
            json={"provider": provider},
        )

    async def handle_callback(
        self,
        provider: OAuthProvider,
        code: str,
        state: str | None = None,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        """Handle OAuth callback (exchange code for tokens)."""
        path = (
            "/api/v1/auth/oauth/oidc/callback"
            if provider == "oidc"
            else f"/api/v1/auth/oauth/{provider}/callback"
        )
        params: dict[str, Any] = {"code": code}
        if state:
            params["state"] = state
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        return await self._client.request("GET", path, params=params)
