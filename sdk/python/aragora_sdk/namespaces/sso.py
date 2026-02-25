"""
SSO (Single Sign-On) Namespace API

Provides methods for enterprise SSO authentication:
- Check SSO configuration status
- Initiate SSO login flow
- Handle SSO callback from identity provider
- Logout from SSO session
- Retrieve SAML/OIDC provider metadata
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SSOAPI:
    """
    Synchronous SSO API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> status = client.sso.get_status()
        >>> if status.get("enabled"):
        ...     login = client.sso.login()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_status(self) -> dict[str, Any]:
        """Get SSO configuration status."""
        return self._client.request("GET", "/api/v2/sso/status")

    def login(self) -> dict[str, Any]:
        """Initiate SSO login flow."""
        return self._client.request("GET", "/api/v2/sso/login")

    def callback(self) -> dict[str, Any]:
        """Handle SSO callback (GET)."""
        return self._client.request("GET", "/api/v2/sso/callback")

    def callback_post(self, **kwargs: Any) -> dict[str, Any]:
        """Handle SSO callback (POST)."""
        return self._client.request("POST", "/api/v2/sso/callback", json=kwargs)

    def logout(self) -> dict[str, Any]:
        """Logout from SSO session."""
        return self._client.request("GET", "/api/v2/sso/logout")

    def get_metadata(self) -> dict[str, Any]:
        """Get SSO provider metadata."""
        return self._client.request("GET", "/api/v2/sso/metadata")

    def sso_login(self) -> dict[str, Any]:
        """Initiate SSO login via auth endpoint.

        Returns:
            Dict with SSO login redirect URL and session data.
        """
        return self._client.request("GET", "/auth/sso/login")

    def auth_callback(self) -> dict[str, Any]:
        """Handle SSO callback via auth endpoint (GET)."""
        return self._client.request("GET", "/auth/sso/callback")

    def auth_callback_post(self, **kwargs: Any) -> dict[str, Any]:
        """Handle SSO callback via auth endpoint (POST)."""
        return self._client.request("POST", "/auth/sso/callback", json=kwargs)

    def auth_logout(self) -> dict[str, Any]:
        """Logout from SSO via auth endpoint."""
        return self._client.request("GET", "/auth/sso/logout")

    def auth_metadata(self) -> dict[str, Any]:
        """Get SSO provider metadata via auth endpoint."""
        return self._client.request("GET", "/auth/sso/metadata")

    def auth_status(self) -> dict[str, Any]:
        """Get SSO configuration status via auth endpoint."""
        return self._client.request("GET", "/auth/sso/status")


class AsyncSSOAPI:
    """
    Asynchronous SSO API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.sso.get_status()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_status(self) -> dict[str, Any]:
        """Get SSO configuration status."""
        return await self._client.request("GET", "/api/v2/sso/status")

    async def login(self) -> dict[str, Any]:
        """Initiate SSO login flow."""
        return await self._client.request("GET", "/api/v2/sso/login")

    async def callback(self) -> dict[str, Any]:
        """Handle SSO callback (GET)."""
        return await self._client.request("GET", "/api/v2/sso/callback")

    async def callback_post(self, **kwargs: Any) -> dict[str, Any]:
        """Handle SSO callback (POST)."""
        return await self._client.request("POST", "/api/v2/sso/callback", json=kwargs)

    async def logout(self) -> dict[str, Any]:
        """Logout from SSO session."""
        return await self._client.request("GET", "/api/v2/sso/logout")

    async def get_metadata(self) -> dict[str, Any]:
        """Get SSO provider metadata."""
        return await self._client.request("GET", "/api/v2/sso/metadata")

    async def sso_login(self) -> dict[str, Any]:
        """Initiate SSO login via auth endpoint."""
        return await self._client.request("GET", "/auth/sso/login")

    async def auth_callback(self) -> dict[str, Any]:
        """Handle SSO callback via auth endpoint (GET)."""
        return await self._client.request("GET", "/auth/sso/callback")

    async def auth_callback_post(self, **kwargs: Any) -> dict[str, Any]:
        """Handle SSO callback via auth endpoint (POST)."""
        return await self._client.request("POST", "/auth/sso/callback", json=kwargs)

    async def auth_logout(self) -> dict[str, Any]:
        """Logout from SSO via auth endpoint."""
        return await self._client.request("GET", "/auth/sso/logout")

    async def auth_metadata(self) -> dict[str, Any]:
        """Get SSO provider metadata via auth endpoint."""
        return await self._client.request("GET", "/auth/sso/metadata")

    async def auth_status(self) -> dict[str, Any]:
        """Get SSO configuration status via auth endpoint."""
        return await self._client.request("GET", "/auth/sso/status")
