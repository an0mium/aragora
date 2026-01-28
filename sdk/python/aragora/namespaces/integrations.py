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
        limit: int = 20,
        offset: int = 0,
        provider: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List integrations (v2).

        Args:
            limit: Maximum number of integrations to return
            offset: Number of integrations to skip
            provider: Filter by provider/type (slack, teams, discord, email)
            status: Filter by status (active, inactive)

        Returns:
            Integration listing with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if provider:
            params["type"] = provider
        if status:
            params["status"] = status

        return self._client._request("GET", "/api/v2/integrations", params=params)

    def get(self, integration_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """
        Get integration status (v2).

        Args:
            integration_id: Integration type (slack, teams, discord, email)
            workspace_id: Optional workspace/tenant ID

        Returns:
            Integration status
        """
        params = {"workspace_id": workspace_id} if workspace_id else None
        return self._client._request(
            "GET",
            f"/api/v2/integrations/{integration_id}",
            params=params,
        )

    def create(self, provider: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Configure a v1 integration by type.

        Args:
            provider: Integration type (slack, teams, discord, email)
            config: Provider-specific configuration

        Returns:
            Configuration result
        """
        return self._client._request(
            "PUT",
            f"/api/v1/integrations/{provider}",
            json=config,
        )

    def update(
        self,
        integration_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update an integration configuration (v1).

        Args:
            integration_id: Integration type (slack, teams, discord, email)
            config: Updated configuration payload

        Returns:
            Updated integration record
        """
        return self._client._request(
            "PATCH",
            f"/api/v1/integrations/{integration_id}",
            json=config,
        )

    def delete(self, integration_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """
        Delete/disconnect an integration.

        Args:
            integration_id: Integration type
            workspace_id: Optional workspace to disconnect (v2)

        Returns:
            Deletion confirmation
        """
        if workspace_id:
            return self._client._request(
                "DELETE",
                f"/api/v2/integrations/{integration_id}",
                json={"workspace_id": workspace_id},
            )
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
        """Sync is not exposed on the current integrations API."""
        raise NotImplementedError("Integration sync is not exposed via the API.")

    def get_oauth_url(self, provider: str, redirect_uri: str) -> dict[str, Any]:
        """OAuth helper is not exposed via the public integrations API."""
        raise NotImplementedError("Use provider-specific OAuth endpoints.")

    def complete_oauth(self, provider: str, code: str, state: str) -> dict[str, Any]:
        """OAuth helper is not exposed via the public integrations API."""
        raise NotImplementedError("Use provider-specific OAuth endpoints.")

    def list_providers(self) -> dict[str, Any]:
        """
        List available integration providers.

        Returns:
            List of available providers with capabilities
        """
        return self._client._request("GET", "/api/v2/integrations/wizard/providers")


class AsyncIntegrationsAPI:
    """Asynchronous integrations API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
        provider: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List integrations (v2)."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if provider:
            params["type"] = provider
        if status:
            params["status"] = status

        return await self._client._request("GET", "/api/v2/integrations", params=params)

    async def get(self, integration_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """Get integration status (v2)."""
        params = {"workspace_id": workspace_id} if workspace_id else None
        return await self._client._request(
            "GET",
            f"/api/v2/integrations/{integration_id}",
            params=params,
        )

    async def create(self, provider: str, config: dict[str, Any]) -> dict[str, Any]:
        """Configure a v1 integration by type."""
        return await self._client._request(
            "PUT",
            f"/api/v1/integrations/{provider}",
            json=config,
        )

    async def update(self, integration_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Update an integration configuration (v1)."""
        return await self._client._request(
            "PATCH",
            f"/api/v1/integrations/{integration_id}",
            json=config,
        )

    async def delete(self, integration_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """Delete/disconnect an integration."""
        if workspace_id:
            return await self._client._request(
                "DELETE",
                f"/api/v2/integrations/{integration_id}",
                json={"workspace_id": workspace_id},
            )
        return await self._client._request("DELETE", f"/api/v1/integrations/{integration_id}")

    async def test(self, integration_id: str) -> dict[str, Any]:
        """Test an integration connection."""
        return await self._client._request("POST", f"/api/v1/integrations/{integration_id}/test")

    async def sync(self, integration_id: str) -> dict[str, Any]:
        """Sync is not exposed on the current integrations API."""
        raise NotImplementedError("Integration sync is not exposed via the API.")

    async def get_oauth_url(self, provider: str, redirect_uri: str) -> dict[str, Any]:
        """OAuth helper is not exposed via the public integrations API."""
        raise NotImplementedError("Use provider-specific OAuth endpoints.")

    async def complete_oauth(self, provider: str, code: str, state: str) -> dict[str, Any]:
        """OAuth helper is not exposed via the public integrations API."""
        raise NotImplementedError("Use provider-specific OAuth endpoints.")

    async def list_providers(self) -> dict[str, Any]:
        """List available integration providers."""
        return await self._client._request("GET", "/api/v2/integrations/wizard/providers")
