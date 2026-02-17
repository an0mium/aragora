"""
Partner API Namespace.

Provides methods for partner management:
- Partner registration and profile
- API key management
- Usage statistics
- Webhook configuration
- Rate limit information
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PartnerAPI:
    """
    Synchronous Partner API.

    Provides methods for partner registration, API key management,
    and usage tracking.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.partner.register(name="My Company", email="partner@example.com")
        >>> keys = client.partner.list_keys()
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Registration
    # =========================================================================

    def register(self, **kwargs: Any) -> dict[str, Any]:
        """Register as a partner."""
        return self._client.request("POST", "/api/v1/partners/register", json=kwargs)

    def get_profile(self) -> dict[str, Any]:
        """Get partner profile."""
        return self._client.request("GET", "/api/v1/partners/me")

    # =========================================================================
    # API Keys
    # =========================================================================

    def list_keys(self) -> dict[str, Any]:
        """List API keys."""
        return self._client.request("GET", "/api/v1/partners/keys")

    def revoke_key(self, key_id: str) -> dict[str, Any]:
        """Revoke an API key."""
        return self._client.request("DELETE", f"/api/v1/partners/keys/{key_id}")

    # =========================================================================
    # Usage & Limits
    # =========================================================================

    def get_usage(self) -> dict[str, Any]:
        """Get usage statistics."""
        return self._client.request("GET", "/api/v1/partners/usage")

    def get_limits(self) -> dict[str, Any]:
        """Get rate limits."""
        return self._client.request("GET", "/api/v1/partners/limits")

    # =========================================================================
    # Webhooks
    # =========================================================================

    def get_webhooks(self) -> dict[str, Any]:
        """Get webhook configuration."""
        return self._client.request("GET", "/api/v1/partners/webhooks")


class AsyncPartnerAPI:
    """Asynchronous Partner API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def register(self, **kwargs: Any) -> dict[str, Any]:
        """Register as a partner."""
        return await self._client.request("POST", "/api/v1/partners/register", json=kwargs)

    async def get_profile(self) -> dict[str, Any]:
        """Get partner profile."""
        return await self._client.request("GET", "/api/v1/partners/me")

    async def list_keys(self) -> dict[str, Any]:
        """List API keys."""
        return await self._client.request("GET", "/api/v1/partners/keys")

    async def revoke_key(self, key_id: str) -> dict[str, Any]:
        """Revoke an API key."""
        return await self._client.request("DELETE", f"/api/v1/partners/keys/{key_id}")

    async def get_usage(self) -> dict[str, Any]:
        """Get usage statistics."""
        return await self._client.request("GET", "/api/v1/partners/usage")

    async def get_limits(self) -> dict[str, Any]:
        """Get rate limits."""
        return await self._client.request("GET", "/api/v1/partners/limits")

    async def get_webhooks(self) -> dict[str, Any]:
        """Get webhook configuration."""
        return await self._client.request("GET", "/api/v1/partners/webhooks")
