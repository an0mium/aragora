"""
Integrations namespace for external service connections.

Provides API access to manage integrations with external services
like Slack, Discord, GitHub, Jira, and other platforms.
"""

from __future__ import annotations

from typing import Any


class IntegrationsAPI:
    """Synchronous integrations API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List integrations.

        Args:
            limit: Maximum number of integrations to return
            offset: Number of integrations to skip
            provider: Filter by provider (slack, discord, github, etc.)
            status: Filter by status (active, inactive, error)

        Returns:
            List of integration records
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if provider:
            params["provider"] = provider
        if status:
            params["status"] = status

        return self._client._request("GET", "/api/v1/integrations", params=params)

    def get(self, integration_id: str) -> dict[str, Any]:
        """
        Get integration details.

        Args:
            integration_id: Integration identifier

        Returns:
            Integration details
        """
        return self._client._request("GET", f"/api/v1/integrations/{integration_id}")

    def create(
        self,
        provider: str,
        name: str,
        config: dict[str, Any],
        enabled: bool = True,
    ) -> dict[str, Any]:
        """
        Create a new integration.

        Args:
            provider: Integration provider (slack, discord, github, etc.)
            name: Display name for the integration
            config: Provider-specific configuration
            enabled: Whether integration is active

        Returns:
            Created integration record
        """
        return self._client._request(
            "POST",
            "/api/v1/integrations",
            json={
                "provider": provider,
                "name": name,
                "config": config,
                "enabled": enabled,
            },
        )

    def update(
        self,
        integration_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """
        Update an integration.

        Args:
            integration_id: Integration identifier
            name: New display name
            config: Updated configuration
            enabled: Updated enabled status

        Returns:
            Updated integration record
        """
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if config is not None:
            data["config"] = config
        if enabled is not None:
            data["enabled"] = enabled

        return self._client._request("PATCH", f"/api/v1/integrations/{integration_id}", json=data)

    def delete(self, integration_id: str) -> dict[str, Any]:
        """
        Delete an integration.

        Args:
            integration_id: Integration identifier

        Returns:
            Deletion confirmation
        """
        return self._client._request("DELETE", f"/api/v1/integrations/{integration_id}")

    def test(self, integration_id: str) -> dict[str, Any]:
        """
        Test an integration connection.

        Args:
            integration_id: Integration identifier

        Returns:
            Test results
        """
        return self._client._request("POST", f"/api/v1/integrations/{integration_id}/test")

    def sync(self, integration_id: str) -> dict[str, Any]:
        """
        Trigger a sync for an integration.

        Args:
            integration_id: Integration identifier

        Returns:
            Sync status
        """
        return self._client._request("POST", f"/api/v1/integrations/{integration_id}/sync")

    def get_oauth_url(self, provider: str, redirect_uri: str) -> dict[str, Any]:
        """
        Get OAuth authorization URL for a provider.

        Args:
            provider: Integration provider
            redirect_uri: OAuth redirect URI

        Returns:
            OAuth authorization URL and state
        """
        return self._client._request(
            "POST",
            f"/api/v1/integrations/oauth/{provider}/authorize",
            json={"redirect_uri": redirect_uri},
        )

    def complete_oauth(self, provider: str, code: str, state: str) -> dict[str, Any]:
        """
        Complete OAuth flow for a provider.

        Args:
            provider: Integration provider
            code: OAuth authorization code
            state: OAuth state parameter

        Returns:
            Created integration
        """
        return self._client._request(
            "POST",
            f"/api/v1/integrations/oauth/{provider}/callback",
            json={"code": code, "state": state},
        )

    def list_providers(self) -> list[dict[str, Any]]:
        """
        List available integration providers.

        Returns:
            List of available providers with capabilities
        """
        return self._client._request("GET", "/api/v1/integrations/providers")


class AsyncIntegrationsAPI:
    """Asynchronous integrations API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List integrations."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if provider:
            params["provider"] = provider
        if status:
            params["status"] = status

        return await self._client._request("GET", "/api/v1/integrations", params=params)

    async def get(self, integration_id: str) -> dict[str, Any]:
        """Get integration details."""
        return await self._client._request("GET", f"/api/v1/integrations/{integration_id}")

    async def create(
        self,
        provider: str,
        name: str,
        config: dict[str, Any],
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Create a new integration."""
        return await self._client._request(
            "POST",
            "/api/v1/integrations",
            json={
                "provider": provider,
                "name": name,
                "config": config,
                "enabled": enabled,
            },
        )

    async def update(
        self,
        integration_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Update an integration."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if config is not None:
            data["config"] = config
        if enabled is not None:
            data["enabled"] = enabled

        return await self._client._request(
            "PATCH", f"/api/v1/integrations/{integration_id}", json=data
        )

    async def delete(self, integration_id: str) -> dict[str, Any]:
        """Delete an integration."""
        return await self._client._request("DELETE", f"/api/v1/integrations/{integration_id}")

    async def test(self, integration_id: str) -> dict[str, Any]:
        """Test an integration connection."""
        return await self._client._request("POST", f"/api/v1/integrations/{integration_id}/test")

    async def sync(self, integration_id: str) -> dict[str, Any]:
        """Trigger a sync for an integration."""
        return await self._client._request("POST", f"/api/v1/integrations/{integration_id}/sync")

    async def get_oauth_url(self, provider: str, redirect_uri: str) -> dict[str, Any]:
        """Get OAuth authorization URL for a provider."""
        return await self._client._request(
            "POST",
            f"/api/v1/integrations/oauth/{provider}/authorize",
            json={"redirect_uri": redirect_uri},
        )

    async def complete_oauth(self, provider: str, code: str, state: str) -> dict[str, Any]:
        """Complete OAuth flow for a provider."""
        return await self._client._request(
            "POST",
            f"/api/v1/integrations/oauth/{provider}/callback",
            json={"code": code, "state": state},
        )

    async def list_providers(self) -> list[dict[str, Any]]:
        """List available integration providers."""
        return await self._client._request("GET", "/api/v1/integrations/providers")
