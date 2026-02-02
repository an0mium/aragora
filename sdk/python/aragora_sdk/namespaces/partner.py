"""
Partner API Namespace.

Provides methods for partner management:
- Partner registration and profile
- API key management
- Usage statistics
- Webhook configuration
- Rate limit information

Endpoints:
    POST   /api/v1/partners/register     - Register as partner
    GET    /api/v1/partners/me           - Get partner profile
    POST   /api/v1/partners/keys         - Create API key
    GET    /api/v1/partners/keys         - List API keys
    DELETE /api/v1/partners/keys/{id}    - Revoke API key
    GET    /api/v1/partners/usage        - Get usage statistics
    POST   /api/v1/partners/webhooks     - Configure webhook
    GET    /api/v1/partners/limits       - Get rate limits
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
        >>> # Register as a partner
        >>> result = client.partner.register(
        ...     name="My Company",
        ...     email="partner@example.com",
        ...     company="My Company Inc",
        ... )
        >>> # Create an API key
        >>> key = client.partner.create_api_key(
        ...     name="Production Key",
        ...     scopes=["debates:read", "debates:write"],
        ... )
        >>> print(f"API Key: {key['key']}")  # Only shown once!
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Registration
    # =========================================================================

    def register(
        self,
        name: str,
        email: str,
        company: str | None = None,
    ) -> dict[str, Any]:
        """
        Register as a partner.

        Args:
            name: Partner name.
            email: Contact email.
            company: Optional company name.

        Returns:
            Dict with partner_id, status, referral_code.
        """
        data: dict[str, Any] = {"name": name, "email": email}
        if company:
            data["company"] = company

        return self._client.request("POST", "/api/v1/partners/register", json=data)

    def get_profile(self) -> dict[str, Any]:
        """
        Get current partner profile.

        Requires X-Partner-ID header.

        Returns:
            Dict with partner stats and profile.
        """
        return self._client.request("GET", "/api/v1/partners/me")

    # =========================================================================
    # API Keys
    # =========================================================================

    def create_api_key(
        self,
        name: str = "API Key",
        scopes: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a new API key.

        Args:
            name: Key name.
            scopes: Optional list of scopes.
            expires_in_days: Optional expiration days.

        Returns:
            Dict with key_id, key (only shown once!), scopes.
        """
        data: dict[str, Any] = {"name": name}
        if scopes:
            data["scopes"] = scopes
        if expires_in_days:
            data["expires_in_days"] = expires_in_days

        return self._client.request("POST", "/api/v1/partners/keys", json=data)

    def list_api_keys(self) -> dict[str, Any]:
        """
        List all API keys.

        Returns:
            Dict with keys array and counts.
        """
        return self._client.request("GET", "/api/v1/partners/keys")

    def revoke_api_key(self, key_id: str) -> dict[str, Any]:
        """
        Revoke an API key.

        Args:
            key_id: The key ID to revoke.

        Returns:
            Dict with success message.
        """
        return self._client.request("DELETE", f"/api/v1/partners/keys/{key_id}")

    # =========================================================================
    # Usage
    # =========================================================================

    def get_usage(self, days: int = 30) -> dict[str, Any]:
        """
        Get usage statistics.

        Args:
            days: Number of days to look back (1-365).

        Returns:
            Dict with usage statistics.
        """
        return self._client.request("GET", "/api/v1/partners/usage", params={"days": days})

    # =========================================================================
    # Webhooks
    # =========================================================================

    def configure_webhook(self, url: str) -> dict[str, Any]:
        """
        Configure webhook endpoint.

        Args:
            url: Webhook URL (must be HTTPS).

        Returns:
            Dict with webhook_url and webhook_secret (only shown once!).
        """
        return self._client.request("POST", "/api/v1/partners/webhooks", json={"url": url})

    # =========================================================================
    # Limits
    # =========================================================================

    def get_limits(self) -> dict[str, Any]:
        """
        Get rate limits for partner tier.

        Returns:
            Dict with tier, limits, current usage, allowed status.
        """
        return self._client.request("GET", "/api/v1/partners/limits")


class AsyncPartnerAPI:
    """Asynchronous Partner API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Registration
    # =========================================================================

    async def register(
        self,
        name: str,
        email: str,
        company: str | None = None,
    ) -> dict[str, Any]:
        """Register as a partner."""
        data: dict[str, Any] = {"name": name, "email": email}
        if company:
            data["company"] = company

        return await self._client.request("POST", "/api/v1/partners/register", json=data)

    async def get_profile(self) -> dict[str, Any]:
        """Get current partner profile."""
        return await self._client.request("GET", "/api/v1/partners/me")

    # =========================================================================
    # API Keys
    # =========================================================================

    async def create_api_key(
        self,
        name: str = "API Key",
        scopes: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> dict[str, Any]:
        """Create a new API key."""
        data: dict[str, Any] = {"name": name}
        if scopes:
            data["scopes"] = scopes
        if expires_in_days:
            data["expires_in_days"] = expires_in_days

        return await self._client.request("POST", "/api/v1/partners/keys", json=data)

    async def list_api_keys(self) -> dict[str, Any]:
        """List all API keys."""
        return await self._client.request("GET", "/api/v1/partners/keys")

    async def revoke_api_key(self, key_id: str) -> dict[str, Any]:
        """Revoke an API key."""
        return await self._client.request("DELETE", f"/api/v1/partners/keys/{key_id}")

    # =========================================================================
    # Usage
    # =========================================================================

    async def get_usage(self, days: int = 30) -> dict[str, Any]:
        """Get usage statistics."""
        return await self._client.request("GET", "/api/v1/partners/usage", params={"days": days})

    # =========================================================================
    # Webhooks
    # =========================================================================

    async def configure_webhook(self, url: str) -> dict[str, Any]:
        """Configure webhook endpoint."""
        return await self._client.request("POST", "/api/v1/partners/webhooks", json={"url": url})

    # =========================================================================
    # Limits
    # =========================================================================

    async def get_limits(self) -> dict[str, Any]:
        """Get rate limits for partner tier."""
        return await self._client.request("GET", "/api/v1/partners/limits")
